"""Run the ticket-triage example for real and save a full trace (used by readsigil.com/lab demo)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promptlab.core import *
from promptlab.cli import _model
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "ticket_triage.json")))
samples = [Sample(s["inputs"], s.get("expected")) for s in cfg["samples"]]
target, opt, judge = _model(cfg["target"]), _model(cfg["optimizer"]), _model(cfg["judge"])
ev = Evaluator(rubric=cfg["evaluator"]["rubric"], judge=judge)
rounds = []
def cb(h):
    r = h["result"]
    rounds.append({"round": h["round"], "score": r.mean, "template": h["template"], "diagnosis": h["diagnosis"],
                   "latency_s": r.latency_s, "cost_usd": r.cost_usd, "tokens": r.tokens, "samples": r.samples})
    print("round", h["round"], round(r.mean, 3), flush=True)
out = optimize(cfg["template"], samples, ev, target, opt, cfg["max_rounds"], cfg["patience"], on_round=cb)
cmp = compare(out["baseline"], out["best"], [_model(m) for m in cfg["compare"]], samples, ev)
res = {"ran_at": time.strftime("%Y-%m-%d %H:%M %Z"), "provider": "groq", "target": target.name, "optimizer": opt.name,
       "judge": judge.name, "rubric": cfg["evaluator"]["rubric"], "baseline": out["baseline"], "best": out["best"],
       "baseline_score": out["baseline_score"], "best_score": out["best_score"], "rounds": rounds,
       "diff": diff(out["baseline"], out["best"]),
       "compare": [{"model": c["model"], "baseline": c["baseline"].mean, "optimized": c["optimized"].mean,
                    "baseline_latency_s": c["baseline"].latency_s, "optimized_latency_s": c["optimized"].latency_s,
                    "baseline_cost_usd": c["baseline"].cost_usd, "optimized_cost_usd": c["optimized"].cost_usd,
                    "baseline_samples": c["baseline"].samples, "optimized_samples": c["optimized"].samples} for c in cmp]}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "demo-run.json"), "w"), indent=1)
for c in res["compare"]: print(c["model"], round(c["baseline"], 3), "->", round(c["optimized"], 3))
