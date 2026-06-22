from eval.text_metrics import exact_match, normalize_answer, token_f1


def test_normalize_answer_strips_articles_punctuation_and_case():
    assert normalize_answer("The  Quick, Brown FOX!") == "quick brown fox"


def test_exact_match_ignores_articles_and_punctuation():
    assert exact_match("Fahrenheit 451.", "fahrenheit 451") == 1.0
    assert exact_match("Swagger UI and ReDoc", "Swagger UI") == 0.0


def test_token_f1_partial_overlap_is_between_zero_and_one():
    pred = "Swagger UI and ReDoc"
    gold = "Swagger UI and ReDoc are included by default"
    score = token_f1(pred, gold)
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("the capital of France", "Fahrenheit 451 novel") == 0.0


def test_token_f1_exact_overlap_is_one():
    assert token_f1("502 Bad Gateway", "a 502 bad gateway") == 1.0


def test_metrics_handle_empty_strings():
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert exact_match("", "") == 1.0
