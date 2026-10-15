"""promptlab CLI.  promptlab run config.json [--out report.json]"""
from __future__ import annotations
import argparse, json, os, sys
from .core import (OpenAICompatModel, Sample, Evaluator, optimize, compare, diff, PROVIDERS)


def _model(spec: dict) -> OpenAICompatModel:
    prov = spec.get("provider", "groq")
    key = os.environ.get(spec.get("api_key_env") or f"{prov.upper()}_API_KEY", "")
    if not key:
        sys.exit(f"missing API key env var for provider {prov}")
    return OpenAICompatModel(spec["model"], key, spec.get("base_url", PROVIDERS.get(prov, prov)),
                             spec.get("price_in", 0), spec.get("price_out", 0))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promptlab", description="Open, model-agnostic prompt optimizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="optimize a prompt from a JSON config")
    r.add_argument("config"); r.add_argument("--out", default="promptlab-report.json")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    samples = [Sample(s["inputs"], s.get("expected")) for s in cfg["samples"]]
    target = _model(cfg["target"]); opt = _model(cfg.get("optimizer", cfg["target"]))
    judge = _model(cfg.get("judge", cfg.get("optimizer", cfg["target"])))
    ev = cfg["evaluator"]
    evaluator = Evaluator.steering(ev["steering"], judge) if "steering" in ev else Evaluator(rubric=ev["rubric"], judge=judge)

    def show(h):
        res = h["result"]
        print(f"round {h['round']}: mean {res.mean:.3f}  latency {res.latency_s:.2f}s  tokens {res.tokens}", flush=True)

    out = optimize(cfg["template"], samples, evaluator, target, opt, cfg.get("max_rounds", 5),
                   cfg.get("patience", 2), on_round=show)
    cmp_models = [_model(m) for m in cfg.get("compare", [])]
    if cmp_models:
        rows = compare(out["baseline"], out["best"], cmp_models, samples, evaluator)
        out["compare"] = [{"model": x["model"], "baseline": x["baseline"].mean, "optimized": x["optimized"].mean,
                           "baseline_latency_s": x["baseline"].latency_s, "optimized_latency_s": x["optimized"].latency_s,
                           "baseline_cost_usd": x["baseline"].cost_usd, "optimized_cost_usd": x["optimized"].cost_usd}
                          for x in rows]
        for c in out["compare"]:
            print(f"{c['model']}: {c['baseline']:.3f} -> {c['optimized']:.3f}")
    out["diff"] = diff(out["baseline"], out["best"])
    json.dump(out, open(a.out, "w"), indent=2, default=lambda o: o.__dict__)
    print(f"\nbaseline {out['baseline_score']:.3f} -> best {out['best_score']:.3f}\n\n{out['diff']}\n\nreport: {a.out}")


if __name__ == "__main__":
    main()
