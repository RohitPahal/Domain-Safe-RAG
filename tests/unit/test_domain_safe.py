import pytest

from rag_thesis.guard.domain_safe import extract_critical_tokens, validate_and_repair_rewrite
from rag_thesis.llm.errors import LLMAuthError, LLMTransientError
from rag_thesis.llm.testing import FakeChatModel

REQUIRED_REPORT_FIELDS = {
    "is_safe",
    "original_critical_tokens",
    "rewrite_critical_tokens",
    "missing_tokens",
    "added_tokens",
    "negation_added",
    "repair_attempted",
    "final_action",
    "reason_codes",
    "initial_assessment",  # why the candidate was flagged / why a repair failed
    "repair_error",  # why the candidate was flagged / why a repair failed
}


def test_safe_rewrite_passes_without_repair() -> None:
    original = "What is the role of IL-6 in sepsis?"
    rewrite = "Explain the role of IL-6 in sepsis."

    final_query, report = validate_and_repair_rewrite(original, rewrite)

    assert final_query == rewrite
    assert report["is_safe"] is True
    assert report["repair_attempted"] is False
    assert report["final_action"] == "pass"
    assert report["missing_tokens"] == []
    assert report["added_tokens"] == []
    assert "IL-6" in report["original_critical_tokens"]
    assert set(report) == REQUIRED_REPORT_FIELDS


def test_critical_token_extraction_covers_required_token_types() -> None:
    query = (
        'Compare "Tumor Necrosis Factor" with Acute Lymphoblastic Leukemia markers '
        "BRCA1, CD4, 5-FU, TNF-alpha, HLA-DR, HIV-1/2, p53, 12.5%, 2024-01-31, and 42."
    )

    tokens = [token.text for token in extract_critical_tokens(query)]

    assert '"Tumor Necrosis Factor"' in tokens
    assert "Acute Lymphoblastic Leukemia" in tokens
    assert "BRCA1" in tokens
    assert "CD4" in tokens
    assert "5-FU" in tokens
    assert "TNF-alpha" in tokens
    assert "HLA-DR" in tokens
    assert "HIV-1/2" in tokens
    assert "p53" in tokens
    assert "12.5%" in tokens
    assert "2024-01-31" in tokens
    assert "42" in tokens


def test_added_number_falls_back_to_original_without_client() -> None:
    original = "What is the function of BRCA1?"
    rewrite = "What is the function of BRCA1 stage 2?"

    final_query, report = validate_and_repair_rewrite(original, rewrite)

    assert final_query == original
    assert report["is_safe"] is False
    assert report["repair_attempted"] is False
    assert report["final_action"] == "fallback_original"
    assert "2" in report["added_tokens"]
    assert "ADDED_SENSITIVE_TOKENS" in report["reason_codes"]
    assert "REPAIR_UNAVAILABLE" in report["reason_codes"]
    assert set(report) == REQUIRED_REPORT_FIELDS


def test_added_identifier_falls_back_to_original_without_client() -> None:
    original = "What is the function of BRCA1?"
    rewrite = "What is the function of BRCA1 and BRCA2?"

    final_query, report = validate_and_repair_rewrite(original, rewrite)

    assert final_query == original
    assert report["is_safe"] is False
    assert report["final_action"] == "fallback_original"
    assert "BRCA2" in report["added_tokens"]
    assert "ADDED_SENSITIVE_TOKENS" in report["reason_codes"]


def test_missing_abbreviation_triggers_single_repair_attempt_and_succeeds() -> None:
    original = "What is the role of BRCA1 in breast cancer?"
    rewrite = "What is the role in breast cancer?"
    client = FakeChatModel(["What is the role of BRCA1 in breast cancer?"])

    final_query, report = validate_and_repair_rewrite(original, rewrite, repair_model=client)

    assert final_query == "What is the role of BRCA1 in breast cancer?"
    assert report["is_safe"] is True
    assert report["repair_attempted"] is True
    assert report["final_action"] == "repaired"
    assert report["missing_tokens"] == []
    assert len(client.requests) == 1


def test_repair_is_limited_to_one_attempt_and_rejects_if_still_unsafe() -> None:
    original = "What is the role of CD4 in immunity?"
    rewrite = "What is the role in immunity?"
    client = FakeChatModel(["What is the role in immunity?"])

    final_query, report = validate_and_repair_rewrite(original, rewrite, repair_model=client)

    assert final_query == original
    assert report["is_safe"] is False
    assert report["repair_attempted"] is True
    assert report["final_action"] == "rejected"
    assert "CD4" in report["missing_tokens"]
    assert "REPAIR_UNSAFE" in report["reason_codes"]
    assert len(client.requests) == 1


def test_client_failure_falls_back_to_original_gracefully() -> None:
    original = 'What does "TNF-alpha" indicate in sepsis?'
    rewrite = "What does TNF indicate in sepsis?"
    client = FakeChatModel([LLMAuthError("Set the environment variable GROQ_API_KEY to call groq.")])

    final_query, report = validate_and_repair_rewrite(original, rewrite, repair_model=client)

    assert final_query == original
    assert report["is_safe"] is False
    assert report["repair_attempted"] is True
    assert report["final_action"] == "fallback_original"
    assert "REPAIR_FAILED" in report["reason_codes"]
    assert len(client.requests) == 1


def test_added_negation_is_detected_and_falls_back_cleanly() -> None:
    original = "What is the role of IL-6 in sepsis?"
    rewrite = "What is the role of IL-6 in sepsis without infection?"

    final_query, report = validate_and_repair_rewrite(original, rewrite)

    assert final_query == original
    assert report["is_safe"] is False
    assert report["negation_added"] is True
    assert report["final_action"] == "fallback_original"
    assert "NEGATION_ADDED" in report["reason_codes"]


# --- The report keeps why a rewrite was flagged -----------------------------


def _client_replying(text: str) -> FakeChatModel:
    return FakeChatModel([text])


def test_repaired_report_keeps_the_original_failure_reason() -> None:
    original = "What is the role of BRCA1 in breast cancer?"
    client = _client_replying(original)  # the repair restores the dropped token

    _, report = validate_and_repair_rewrite(original, "What is the role in breast cancer?", repair_model=client)

    assert report["final_action"] == "repaired"
    assert report["reason_codes"] == []  # final query is clean (unchanged meaning of this field)
    assert report["initial_assessment"]["is_safe"] is False
    assert report["initial_assessment"]["reason_codes"] == ["MISSING_CRITICAL_TOKENS"]
    assert report["initial_assessment"]["missing_tokens"] == ["BRCA1"]
    assert report["repair_error"] is None


def test_rejected_report_keeps_the_original_failure_reason() -> None:
    client = _client_replying("What is the role in immunity?")  # repair still drops CD4
    _, report = validate_and_repair_rewrite(
        "What is the role of CD4 in immunity?", "What is the role in immunity?", repair_model=client
    )
    assert report["final_action"] == "rejected"
    assert report["initial_assessment"]["missing_tokens"] == ["CD4"]


def test_passed_report_initial_assessment_matches_the_final_one() -> None:
    _, report = validate_and_repair_rewrite("What is IL-6?", "IL-6 function", repair_model=None)
    assert report["final_action"] == "pass"
    assert report["initial_assessment"] == {
        "is_safe": True,
        "reason_codes": [],
        "missing_tokens": [],
        "added_tokens": [],
        "negation_added": False,
    }


def test_repair_failure_keeps_the_error_and_logs_it(caplog) -> None:
    client = FakeChatModel([LLMTransientError("provider unavailable")])
    caplog.set_level("WARNING")

    final_query, report = validate_and_repair_rewrite(
        "What is the role of CD4 in immunity?", "What is the role in immunity?", repair_model=client
    )

    assert final_query == "What is the role of CD4 in immunity?"
    assert report["final_action"] == "fallback_original"
    assert report["repair_error"] == "LLMTransientError: provider unavailable"
    assert "Guard repair failed" in caplog.text


def test_repair_model_receives_the_versioned_repair_prompt_as_plain_text_request() -> None:
    from rag_thesis.prompts.repair import build_repair_messages

    original = "What is the role of BRCA1 in breast cancer?"
    rewrite = "What is the role in breast cancer?"
    model = FakeChatModel([original])
    validate_and_repair_rewrite(original, rewrite, repair_model=model)

    (request,) = model.requests
    assert request.json_mode is False  # the repair reply is plain text
    assert list(request.messages) == build_repair_messages(
        original_query=original, candidate_rewrite=rewrite, critical_tokens=["BRCA1"]
    )


def test_an_empty_repair_reply_falls_back_with_its_reason() -> None:
    _, report = validate_and_repair_rewrite(
        "What is the role of CD4 in immunity?", "What is the role in immunity?", repair_model=FakeChatModel(["  "])
    )
    assert report["final_action"] == "fallback_original"
    assert report["repair_error"] == "EmptyRepairError: Repair response was empty."


@pytest.mark.parametrize("error", [TypeError("a code bug"), ValueError("a code bug"), ImportError("no SDK")])
def test_unexpected_repair_errors_stop_the_run(error) -> None:
    # Only provider failures and unusable replies fall back to the original; anything else stops the run.
    with pytest.raises(type(error)):
        validate_and_repair_rewrite(
            "What is the role of CD4 in immunity?", "What is the role in immunity?", repair_model=FakeChatModel([error])
        )
