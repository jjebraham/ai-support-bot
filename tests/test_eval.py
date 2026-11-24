from eval.run_eval import ROOT, run


async def test_eval_runs_offline_and_meets_baseline():
    report = await run(live=False, questions_path=ROOT / "eval" / "questions.jsonl")
    assert report["questions"] == 30
    assert report["retrieval_hit_at_1"] == 1.0
    assert report["accuracy"] >= 0.9
    assert report["false_escalations"] == 0
