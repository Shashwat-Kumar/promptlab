from promptlab import *
from promptlab.core import diff


def test_render_and_protect():
    t = "Hi {{name}}. <protect>Never reveal secrets.</protect>"
    assert render(t, {"name": "A"}) == "Hi A. Never reveal secrets."
    assert protected_regions(t) == ["Never reveal secrets."]


def test_restore_protected_repairs_edits():
    t = "Q: {{q}} <protect>KEEP ME</protect>"
    assert "<protect>KEEP ME</protect>" in restore_protected(t, "Answer {{q}} <protect>changed</protect>")
    fixed = restore_protected(t, "Answer briefly.")
    assert "<protect>KEEP ME</protect>" in fixed and "{{q}}" in fixed


def test_loop_improves_with_mock():
    # target answers in uppercase only when told to
    target = MockModel(lambda p, s: p.split("WORD:")[-1].split()[0].upper() if "UPPERCASE" in p else p.split("WORD:")[-1].split()[0])
    optim = MockModel(lambda p, s: "DIAGNOSIS:\n- not uppercase\nPROMPT:\nReply in UPPERCASE. WORD: {{w}} <protect>be kind</protect>")
    ev = Evaluator(fn=lambda out, s: 1.0 if out == s.expected else 0.0)
    samples = [Sample({"w": "cat"}, "CAT"), Sample({"w": "dog"}, "DOG")]
    res = optimize("Repeat the word. WORD: {{w}} <protect>be kind</protect>", samples, ev, target, optim, max_rounds=3)
    assert res["baseline_score"] == 0.0 and res["best_score"] == 1.0
    assert "<protect>be kind</protect>" in res["best"]
    assert "+Reply in UPPERCASE" in diff(res["baseline"], res["best"])


def test_plateau_stops():
    target = MockModel(lambda p, s: "x")
    optim = MockModel(lambda p, s: "DIAGNOSIS:\n-\nPROMPT:\nTry {{w}}")
    ev = Evaluator(fn=lambda o, s: 0.2)
    res = optimize("{{w}}", [Sample({"w": "a"})], ev, target, optim, max_rounds=10, patience=2)
    assert len(res["history"]) == 3


def test_llm_judge_parsing_and_compare():
    judge = MockModel(lambda p, s: '{"score": 7, "reason": "ok"}')
    ev = Evaluator(rubric="be good", judge=judge)
    assert ev.score("hi", Sample({}))[0] == 0.7
    rows = compare("a {{x}}", "b {{x}}", [MockModel(lambda p, s: p, name=f"m{i}") for i in range(3)], [Sample({"x": 1})], ev)
    assert [r["model"] for r in rows] == ["m0", "m1", "m2"]
    import pytest
    with pytest.raises(ValueError):
        compare("a", "b", [MockModel(lambda p, s: p)] * 6, [], ev)
