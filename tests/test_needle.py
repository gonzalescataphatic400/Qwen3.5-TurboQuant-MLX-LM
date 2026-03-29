from turbomlx.eval.needle import build_needle_prompt, insert_needle_into_context, score_needle_answer


def test_needle_prompt_does_not_leak_answer_outside_context():
    context = "line one\nline two\nline three"
    needle = "secret launch code 42"
    haystack = insert_needle_into_context(context, needle, insertion_depth_pct=50, seed=7)
    prompt = build_needle_prompt(haystack, "What hidden statement appears in the context?")

    assert "Hidden fact" not in prompt
    assert prompt.count(needle) == 1
    assert "Question: What hidden statement appears in the context?" in prompt


def test_needle_score_uses_exact_normalized_match_as_primary_signal():
    score = score_needle_answer("  Secret   Launch Code 42 ", "secret launch code 42")
    assert score["exact_match"] == 1.0
    assert score["substring_match"] == 1.0
