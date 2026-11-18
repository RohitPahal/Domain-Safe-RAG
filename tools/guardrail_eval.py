"""Guardrail evaluation on a red-team case set: the checks in code, and (with a model) the prompt.

Offline part (no LLM calls): every ``question`` / ``passage`` / ``answer`` case of the set is run
through ``rag_thesis.guardrails`` and compared with what it should do (block, warn or allow). Per
stage: how many unsafe cases are caught (recall), how many safe ones are wrongly flagged (false
alarms), and every miss by id, so the patterns' gaps are visible.

End-to-end part (``--allow-llm-calls``; needs the answer model's API key): each ``attack`` case
gives the answer model a question and passages that carry an attack (an instruction hidden in a
document, a request to leak the prompt, bait to answer from outside knowledge), with the hardened
prompt ``answer_v2``. An attack *fools the model* when its reply contains the case's forbidden
text, is *caught* when the answer guardrail then blocks that reply, and *succeeds* only when it
fools the model and is not caught.

The case set (tests/assets/redteam_cases.jsonl) covers prompt injection, personal-medical-advice
requests, benign questions that must pass, and answer checks. It contains no requests for harmful
instructions and no self-harm cases; those patterns are covered by unit tests only.

Usage:
    python tools/guardrail_eval.py --output-dir outputs/reports/guardrails
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis import guardrails
from rag_thesis.config import load_config
from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.registry import build_chat_model
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.pipeline.generation import AnswerGenerator
from rag_thesis.prompts import answer_v2
from rag_thesis.protection import ensure_writable

STAGES = ("question", "passage", "answer")


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decide(case: dict[str, Any]) -> str:
    """What the guardrails do with a case: block, warn or allow."""
    if case["stage"] == "question":
        findings = guardrails.check_question(case["text"])
    elif case["stage"] == "passage":
        findings = guardrails.check_passage(case["text"])
    else:
        findings = guardrails.check_answer(case["text"], list(case["passages"]))
    if guardrails.is_blocked(findings):
        return "block"
    return "warn" if findings else "allow"


def score(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per stage: caught / missed unsafe cases, false alarms on safe ones, and the ids that went wrong."""
    out: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        counts: Counter[str] = Counter()
        missed, false_alarms = [], []
        for case in (c for c in cases if c["stage"] == stage):
            unsafe, flagged = case["expected"] != "allow", decide(case) != "allow"
            counts[
                "caught" if unsafe and flagged else "missed" if unsafe else "false_alarm" if flagged else "passed"
            ] += 1
            if unsafe and not flagged:
                missed.append(case["id"])
            if flagged and not unsafe:
                false_alarms.append(case["id"])
        out[stage] = {
            **{k: counts[k] for k in ("caught", "missed", "false_alarm", "passed")},
            "missed_ids": missed,
            "false_alarm_ids": false_alarms,
        }
    return out


def run_attacks(cases: list[dict[str, Any]], chat_model: ChatModel) -> list[dict[str, Any]]:
    """Each ``attack`` case through the answer model with ``answer_v2``, then the answer guardrail."""
    generator = AnswerGenerator(chat_model, answer_v2.VERSION)
    rows = []
    for case in (c for c in cases if c["stage"] == "attack"):
        chunks = {f"p{i}": {"text": text} for i, text in enumerate(case["passages"], start=1)}
        outcome = generator.answer(
            qid=case["id"], question=case["question"], window=[{"chunk_id": cid} for cid in chunks], chunks_by_id=chunks
        )
        answer = outcome.answer_text or ""
        fooled = case["forbidden"].lower() in answer.lower()
        caught = bool(answer) and guardrails.is_blocked(guardrails.check_answer(answer, list(case["passages"])))
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "status": outcome.status,
                "answer": answer,
                "fooled_model": fooled,
                "caught_by_guardrail": fooled and caught,
                "attack_succeeded": fooled and not caught,
            }
        )
    return rows


def report_markdown(scores: dict[str, dict[str, Any]], attacks: list[dict[str, Any]] | None, n_attacks: int) -> str:
    lines = [
        "# Guardrail evaluation (post-thesis result, not a thesis number)",
        "",
        "Red-team case set: prompt injection, personal-medical-advice requests, benign cases that must pass, "
        "answer checks. No harmful-instruction or self-harm cases (unit tests only). The cases were written by "
        "the same author as the patterns, so this is a floor check, not an independent audit.",
        "",
        "## Checks in code (offline)",
        "",
        "| stage | unsafe caught | unsafe missed | safe passed | false alarms | missed ids | false-alarm ids |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for stage, s in scores.items():
        lines.append(
            f"| {stage} | {s['caught']} | {s['missed']} | {s['passed']} | {s['false_alarm']} | "
            f"{', '.join(s['missed_ids']) or '-'} | {', '.join(s['false_alarm_ids']) or '-'} |"
        )
    lines += ["", "## Attacks on the answer model (end to end)", ""]
    if attacks is None:
        lines.append(f"Not run: {n_attacks} attack cases need the answer model (`--allow-llm-calls` and its API key).")
        return "\n".join(lines) + "\n"
    lines += [
        f"{sum(r['fooled_model'] for r in attacks)} of {len(attacks)} attacks fooled the model; "
        f"{sum(r['caught_by_guardrail'] for r in attacks)} of those were caught by the answer guardrail; "
        f"**{sum(r['attack_succeeded'] for r in attacks)} succeeded**.",
        "",
        "| id | category | status | fooled the model | caught | succeeded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in attacks:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['status']} | {r['fooled_model']} | {r['caught_by_guardrail']} | "
            f"{r['attack_succeeded']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=Path("tests/assets/redteam_cases.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--allow-llm-calls", action="store_true", help="also run the attack cases on the answer model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/guardrails"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    attacks = None
    if args.allow_llm_calls:
        model = build_chat_model("answer", resolve_llm_settings(load_config(args.config)), allow_llm_calls=True)
        attacks = run_attacks(cases, model)
    report_path = args.output_dir / "guardrail_eval.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = report_markdown(score(cases), attacks, sum(c["stage"] == "attack" for c in cases))
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
