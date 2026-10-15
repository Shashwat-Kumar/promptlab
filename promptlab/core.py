"""Core loop: run -> score -> diagnose -> rewrite -> repeat."""
from __future__ import annotations
import difflib, json, re, time, urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")
PROTECT = re.compile(r"<protect>(.*?)</protect>", re.S)

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def render(template: str, inputs: dict) -> str:
    """Fill {{vars}}; strip <protect> tags (content stays)."""
    out = VAR.sub(lambda m: str(inputs.get(m.group(1), m.group(0))), template)
    return PROTECT.sub(lambda m: m.group(1), out)


def protected_regions(template: str) -> list[str]:
    return PROTECT.findall(template)


def restore_protected(original: str, rewritten: str) -> str:
    """Guarantee every protected region of `original` survives verbatim in `rewritten`.
    Regions are matched by position; missing ones are appended."""
    regions = protected_regions(original)
    found = PROTECT.findall(rewritten)
    it = iter(regions)
    out = PROTECT.sub(lambda m: f"<protect>{next(it, m.group(1))}</protect>", rewritten)
    for r in regions[len(found):]:
        out += f"\n<protect>{r}</protect>"
    # variables must survive too
    for v in set(VAR.findall(original)) - set(VAR.findall(out)):
        out += f"\n{{{{{v}}}}}"
    return out


@dataclass
class Sample:
    inputs: dict
    expected: Optional[str] = None


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class Model:
    name = "model"
    price_in = 0.0   # USD per 1M tokens
    price_out = 0.0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> Completion:
        raise NotImplementedError

    def cost(self, c: Completion) -> float:
        return (c.prompt_tokens * self.price_in + c.completion_tokens * self.price_out) / 1e6


class OpenAICompatModel(Model):
    """Any OpenAI-compatible /chat/completions endpoint (Groq, Gemini, OpenRouter, OpenAI, vLLM...)."""

    def __init__(self, model: str, api_key: str, base_url: str = PROVIDERS["groq"],
                 price_in: float = 0.0, price_out: float = 0.0, max_tokens: int = 1024, retries: int = 4):
        self.name, self.api_key = model, api_key
        self.base_url = PROVIDERS.get(base_url, base_url).rstrip("/")
        self.price_in, self.price_out, self.max_tokens, self.retries = price_in, price_out, max_tokens, retries

    def complete(self, prompt, system=None, temperature=0.0):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({"model": self.name, "messages": msgs, "temperature": temperature,
                           "max_tokens": self.max_tokens}).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "User-Agent": "promptlab/0.1"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < self.retries:
                    time.sleep(2 ** attempt * 3); continue
                raise RuntimeError(f"{self.name}: HTTP {e.code} {e.read()[:300]!r}") from None
        u = d.get("usage") or {}
        return Completion(d["choices"][0]["message"]["content"] or "", time.time() - t0,
                          u.get("prompt_tokens", len(prompt) // 4), u.get("completion_tokens", 0))


class MockModel(Model):
    """Deterministic model for tests: `fn(prompt, system) -> str`."""

    def __init__(self, fn: Callable[[str, Optional[str]], str], name="mock"):
        self.fn, self.name = fn, name

    def complete(self, prompt, system=None, temperature=0.0):
        t = self.fn(prompt, system)
        return Completion(t, 0.001, len(prompt) // 4, len(t) // 4)


JUDGE_SYS = ("You are a strict evaluator. Score the RESPONSE against the CRITERIA (and the EXPECTED answer if given). "
             "Reply with JSON only: {\"score\": <0-10 integer>, \"reason\": \"<one sentence>\"}")


class Evaluator:
    """Three kinds: rubric (LLM judge), function (python), steering (natural-language goal, judged by LLM)."""

    def __init__(self, rubric: str | None = None, fn: Callable[[str, Sample], float] | None = None,
                 judge: Model | None = None):
        if not rubric and not fn:
            raise ValueError("need a rubric/steering text or a python function")
        self.rubric, self.fn, self.judge = rubric, fn, judge

    @classmethod
    def steering(cls, goal: str, judge: Model):
        return cls(rubric=f"The response should satisfy this goal: {goal}", judge=judge)

    def describe(self) -> str:
        return self.rubric or (self.fn.__doc__ or "a programmatic check").strip()

    def score(self, response: str, sample: Sample) -> tuple[float, str]:
        if self.fn:
            v = float(self.fn(response, sample))
            return max(0.0, min(1.0, v)), "function"
        if not self.judge:
            raise ValueError("rubric evaluator needs a judge model")
        p = (f"CRITERIA:\n{self.rubric}\n\nINPUT:\n{json.dumps(sample.inputs, ensure_ascii=False)}\n\n"
             + (f"EXPECTED:\n{sample.expected}\n\n" if sample.expected else "") + f"RESPONSE:\n{response}")
        txt = self.judge.complete(p, JUDGE_SYS).text
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            d = json.loads(m.group(0)); return max(0.0, min(1.0, float(d["score"]) / 10)), str(d.get("reason", ""))
        except Exception:
            n = re.search(r"\d+(\.\d+)?", txt)
            return (min(1.0, float(n.group(0)) / 10) if n else 0.0), txt[:200]


OPT_SYS = """You are a prompt engineer. You improve a prompt TEMPLATE so a model scores higher on an evaluation.
Rules:
- Keep every {{variable}} placeholder exactly as written.
- Text inside <protect>...</protect> must be copied verbatim, tags included. Never edit it.
- Fix the diagnosed failures with clear instructions, output format and constraints; don't overfit to one sample's content.
Reply in exactly this format:
DIAGNOSIS:
<2-4 bullet points on why samples failed>
PROMPT:
<the full new template>"""


class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def rewrite(self, template: str, criteria: str, results: list[dict]) -> tuple[str, str]:
        worst = sorted(results, key=lambda r: r["score"])[:4]
        cases = "\n\n".join(
            f"[score {r['score']:.2f}] INPUT: {json.dumps(r['inputs'], ensure_ascii=False)}\n"
            + (f"EXPECTED: {r['expected']}\n" if r.get("expected") else "")
            + f"OUTPUT: {r['output'][:600]}\nJUDGE: {r['reason']}" for r in worst)
        p = f"CRITERIA:\n{criteria}\n\nCURRENT TEMPLATE:\n{template}\n\nLOWEST-SCORING CASES:\n{cases}"
        txt = self.model.complete(p, OPT_SYS, temperature=0.4).text
        diag, _, new = txt.partition("PROMPT:")
        new = new.strip().strip("`").strip() or template
        diag = diag.replace("DIAGNOSIS:", "").strip()
        return restore_protected(template, new), diag


@dataclass
class RunResult:
    template: str
    model: str
    mean: float
    samples: list[dict]
    latency_s: float
    cost_usd: float
    tokens: int


def evaluate(template: str, model: Model, samples: list[Sample], evaluator: Evaluator) -> RunResult:
    rows, lat, cost, tok = [], 0.0, 0.0, 0
    for s in samples:
        c = model.complete(render(template, s.inputs))
        sc, why = evaluator.score(c.text, s)
        lat += c.latency_s; cost += model.cost(c); tok += c.prompt_tokens + c.completion_tokens
        rows.append({"inputs": s.inputs, "expected": s.expected, "output": c.text, "score": sc,
                     "reason": why, "latency_s": round(c.latency_s, 3)})
    n = max(1, len(rows))
    return RunResult(template, model.name, sum(r["score"] for r in rows) / n, rows, lat / n, cost, tok)


def optimize(template: str, samples: list[Sample], evaluator: Evaluator, target: Model, optimizer: Model,
             max_rounds: int = 5, patience: int = 2, min_gain: float = 0.01,
             on_round: Callable[[dict], None] | None = None) -> dict:
    """Iterate until no gain >= min_gain for `patience` rounds, a perfect score, or max_rounds."""
    opt = Optimizer(optimizer)
    cur = evaluate(template, target, samples, evaluator)
    best, history, stale = cur, [], 0
    history.append({"round": 0, "score": cur.mean, "template": template, "diagnosis": ""})
    if on_round: on_round(history[-1] | {"result": cur})
    for rnd in range(1, max_rounds + 1):
        if best.mean >= 0.999: break
        new_t, diag = opt.rewrite(best.template, evaluator.describe(), best.samples)
        cur = evaluate(new_t, target, samples, evaluator)
        history.append({"round": rnd, "score": cur.mean, "template": new_t, "diagnosis": diag})
        if on_round: on_round(history[-1] | {"result": cur})
        if cur.mean >= best.mean + min_gain:
            best, stale = cur, 0
        else:
            stale += 1
            if stale >= patience: break
    return {"baseline": history[0]["template"], "best": best.template, "best_score": best.mean,
            "baseline_score": history[0]["score"], "history": history}


def compare(baseline: str, optimized: str, models: list[Model], samples: list[Sample], evaluator: Evaluator) -> list[dict]:
    if len(models) > 5:
        raise ValueError("compare supports up to 5 models")
    out = []
    for m in models:
        a, b = evaluate(baseline, m, samples, evaluator), evaluate(optimized, m, samples, evaluator)
        out.append({"model": m.name, "baseline": a, "optimized": b})
    return out


def diff(a: str, b: str) -> str:
    return "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "baseline", "optimized", lineterm=""))
