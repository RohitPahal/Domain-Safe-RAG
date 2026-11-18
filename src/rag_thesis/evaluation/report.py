from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_thesis.evaluation.guard_metrics import guard_markdown, guard_metric_rows
from rag_thesis.io import load_questions
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.protection import ensure_writable
from rag_thesis.schemas import RunLogLine, run_log_usage

QA_METRICS_FILENAME = "qa_metrics.csv"
COST_LATENCY_FILENAME = "cost_latency.csv"
SUMMARY_FILENAME = "final_ablation_summary.md"
GUARD_METRICS_FILENAME = "guard_metrics.csv"
CANONICAL_AUDIT_FILENAME = "audit_manual_sheet.csv"
GENERATED_AUDIT_FILENAME = "audit_manual_sheet.generated.csv"
AUDIT_CONFIG_MAP = {
    "A_hybrid_none_guard_off": ("hybrid", "none", "off"),
    "B_hybrid_multi_guard_on": ("hybrid", "multi", "on"),
}
DEFAULT_PROMPT_PRICE_PER_1K_USD = 0.0
DEFAULT_COMPLETION_PRICE_PER_1K_USD = 0.0


@dataclass(frozen=True, order=True)
class ReportGroupKey:
    retriever_mode: str
    rewrite_mode: str
    guard_mode: str
    k: int
    answer_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever_mode": self.retriever_mode,
            "rewrite_mode": self.rewrite_mode,
            "guard_mode": self.guard_mode,
            "k": self.k,
            "answer_model": self.answer_model,
        }


def load_run_logs(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.jsonl"), key=lambda item: item.name):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = RunLogLine.model_validate(json.loads(line)).model_dump()
                payload["_source_file"] = path.name
                payload["_source_line"] = line_number
                rows.append(payload)
    return rows


def choose_audit_sheet(reports_dir: Path) -> tuple[Path | None, str]:
    canonical = reports_dir / CANONICAL_AUDIT_FILENAME
    generated = reports_dir / GENERATED_AUDIT_FILENAME
    if canonical.exists():
        return canonical, "preferred canonical audit file"
    if generated.exists():
        return generated, "generated audit fallback"
    return None, "no audit sheet present"


def load_audit_rates(audit_path: Path | None) -> dict[tuple[str, str, str], float]:
    if audit_path is None:
        return {}

    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    with audit_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_label = str(row.get("config", "")).strip()
            group_key = AUDIT_CONFIG_MAP.get(config_label)
            if group_key is None:
                continue
            bucket = counts.setdefault(group_key, {"total": 0, "evidence_found": 0})
            bucket["total"] += 1
            if _parse_boolish(row.get("evidence_found")):
                bucket["evidence_found"] += 1

    rates: dict[tuple[str, str, str], float] = {}
    for group_key, stats in counts.items():
        total = stats["total"]
        rates[group_key] = 0.0 if total == 0 else stats["evidence_found"] / float(total)
    return rates


def build_group_key(run_row: dict[str, Any]) -> ReportGroupKey:
    snapshot = dict(run_row.get("config_snapshot", {}))
    retriever_mode = str(snapshot.get("retriever_mode", "")).strip() or "unknown"
    rewrite_mode = str(snapshot.get("rewrite_mode", "")).strip() or "unknown"
    guard_mode = str(snapshot.get("guard_mode", "")).strip() or "unknown"
    k = _coerce_int(snapshot.get("k", snapshot.get("retrieve_k", 0)))
    answer_model = _answer_model_name(snapshot)
    return ReportGroupKey(
        retriever_mode=retriever_mode,
        rewrite_mode=rewrite_mode,
        guard_mode=guard_mode,
        k=k,
        answer_model=answer_model or "unknown",
    )


def citation_is_valid(run_row: dict[str, Any]) -> bool:
    retrieved_chunk_ids = {
        str(result.get("chunk_id")).strip()
        for result in run_row.get("retrieval_results", [])
        if str(result.get("chunk_id", "")).strip()
    }
    citations = [str(citation).strip() for citation in run_row.get("citations", []) if str(citation).strip()]
    return set(citations).issubset(retrieved_chunk_ids)


def aggregate_runs(
    run_rows: list[dict[str, Any]],
    *,
    audit_rates: dict[tuple[str, str, str], float],
    prompt_price_per_1k_usd: float,
    completion_price_per_1k_usd: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[ReportGroupKey, list[dict[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault(build_group_key(row), []).append(row)

    qa_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        rows = grouped[group_key]
        total_questions_attempted = len(rows)
        answered_rows = [row for row in rows if _is_answered(row)]
        total_answered = len(answered_rows)
        unanswered_rate = (
            0.0
            if total_questions_attempted == 0
            else (total_questions_attempted - total_answered) / float(total_questions_attempted)
        )
        citation_valid_rate = (
            0.0
            if total_answered == 0
            else sum(1 for row in answered_rows if citation_is_valid(row)) / float(total_answered)
        )

        latencies = [_coerce_float(row.get("latency_ms")) for row in rows]
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        for row in rows:
            usage = run_log_usage(row)
            prompt_tokens = _coerce_int(usage.get("prompt_tokens", 0))
            completion_tokens = _coerce_int(usage.get("completion_tokens", 0))
            row_total_tokens = _coerce_int(usage.get("total_tokens", prompt_tokens + completion_tokens))
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += row_total_tokens

        estimated_cost_usd = (total_prompt_tokens / 1000.0) * prompt_price_per_1k_usd + (
            total_completion_tokens / 1000.0
        ) * completion_price_per_1k_usd
        avg_tokens_per_question = (
            0.0 if total_questions_attempted == 0 else total_tokens / float(total_questions_attempted)
        )

        audit_key = (group_key.retriever_mode, group_key.rewrite_mode, group_key.guard_mode)
        evidence_found_rate = audit_rates.get(audit_key)

        qa_row = {
            **group_key.to_dict(),
            "total_questions_attempted": total_questions_attempted,
            "total_answered": total_answered,
            "unanswered_rate": _round_float(unanswered_rate, 6),
            "citation_valid_rate": _round_float(citation_valid_rate, 6),
            "evidence_found_rate": "" if evidence_found_rate is None else _round_float(evidence_found_rate, 6),
        }
        cost_row = {
            **group_key.to_dict(),
            "total_questions_attempted": total_questions_attempted,
            "avg_latency_ms": _round_float(_mean(latencies), 3),
            "p50_latency_ms": _round_float(_percentile(latencies, 50.0), 3),
            "p95_latency_ms": _round_float(_percentile(latencies, 95.0), 3),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_question": _round_float(avg_tokens_per_question, 3),
            "estimated_cost_usd": _round_float(estimated_cost_usd, 6),
        }
        qa_rows.append(qa_row)
        cost_rows.append(cost_row)

    return qa_rows, cost_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    *,
    qa_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    audit_path: Path | None,
    audit_rule: str,
    prompt_price_per_1k_usd: float,
    completion_price_per_1k_usd: float,
    guard_rows: list[dict[str, Any]] | None = None,
) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_description = "none"
    if audit_path is not None:
        audit_description = _display_path(audit_path)

    lines = [
        "# Final Ablation Summary",
        "",
        "This report aggregates existing RAG run logs into thesis-ready QA behavior, latency, and token-cost "
        "summaries.",
        "",
        f"Audit sheet selection rule: {audit_rule}.",
        f"Audit sheet used: `{audit_description}`.",
        f"Token pricing constants: prompt `${prompt_price_per_1k_usd:.6f}` per 1K tokens, completion "
        f"`${completion_price_per_1k_usd:.6f}` per 1K tokens.",
        "",
        "## QA Behavior",
        "",
        "| retriever_mode | rewrite_mode | guard_mode | k | answer_model | total_questions_attempted | total_answered "
        "| unanswered_rate | citation_valid_rate | evidence_found_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in qa_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["retriever_mode"]),
                    str(row["rewrite_mode"]),
                    str(row["guard_mode"]),
                    str(row["k"]),
                    str(row["answer_model"]),
                    str(row["total_questions_attempted"]),
                    str(row["total_answered"]),
                    f"{float(row['unanswered_rate']):.6f}",
                    f"{float(row['citation_valid_rate']):.6f}",
                    "" if row["evidence_found_rate"] == "" else f"{float(row['evidence_found_rate']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cost and Latency",
            "",
            "| retriever_mode | rewrite_mode | guard_mode | k | answer_model | avg_latency_ms | p50_latency_ms | "
            "p95_latency_ms | total_prompt_tokens | total_completion_tokens | total_tokens | avg_tokens_per_question | "
            "estimated_cost_usd |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in cost_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["retriever_mode"]),
                    str(row["rewrite_mode"]),
                    str(row["guard_mode"]),
                    str(row["k"]),
                    str(row["answer_model"]),
                    f"{float(row['avg_latency_ms']):.3f}",
                    f"{float(row['p50_latency_ms']):.3f}",
                    f"{float(row['p95_latency_ms']):.3f}",
                    str(row["total_prompt_tokens"]),
                    str(row["total_completion_tokens"]),
                    str(row["total_tokens"]),
                    f"{float(row['avg_tokens_per_question']):.3f}",
                    f"{float(row['estimated_cost_usd']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(guard_markdown(guard_rows or []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qa_report(
    *,
    runs_dir: Path,
    reports_dir: Path,
    prompt_price_per_1k_usd: float = DEFAULT_PROMPT_PRICE_PER_1K_USD,
    completion_price_per_1k_usd: float = DEFAULT_COMPLETION_PRICE_PER_1K_USD,
    questions_path: Path | None = None,
) -> dict[str, Path | None]:
    """QA, latency/cost and guard reports of every run log under ``runs_dir``. With
    ``questions_path`` (their gold answers), the guard report also counts leaking rewrites."""
    run_rows = load_run_logs(runs_dir)
    if not run_rows:
        raise ValueError(f"No run log rows were found under {runs_dir}.")

    audit_path, audit_rule = choose_audit_sheet(reports_dir)
    audit_rates = load_audit_rates(audit_path)
    qa_rows, cost_rows = aggregate_runs(
        run_rows,
        audit_rates=audit_rates,
        prompt_price_per_1k_usd=prompt_price_per_1k_usd,
        completion_price_per_1k_usd=completion_price_per_1k_usd,
    )

    answers_by_qid = None
    if questions_path is not None:
        answers_by_qid = {
            str(q["qid"]): [str(a["text"]) for a in q.get("answers", [])] for q in load_questions(questions_path)
        }
    guard_rows = guard_metric_rows(run_rows, build_group_key, answers_by_qid)

    qa_metrics_path = reports_dir / QA_METRICS_FILENAME
    cost_latency_path = reports_dir / COST_LATENCY_FILENAME
    summary_path = reports_dir / SUMMARY_FILENAME
    write_csv(
        qa_metrics_path,
        qa_rows,
        [
            "retriever_mode",
            "rewrite_mode",
            "guard_mode",
            "k",
            "answer_model",
            "total_questions_attempted",
            "total_answered",
            "unanswered_rate",
            "citation_valid_rate",
            "evidence_found_rate",
        ],
    )
    write_csv(
        cost_latency_path,
        cost_rows,
        [
            "retriever_mode",
            "rewrite_mode",
            "guard_mode",
            "k",
            "answer_model",
            "total_questions_attempted",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "avg_tokens_per_question",
            "estimated_cost_usd",
        ],
    )
    write_summary(
        summary_path,
        qa_rows=qa_rows,
        cost_rows=cost_rows,
        audit_path=audit_path,
        audit_rule=audit_rule,
        prompt_price_per_1k_usd=prompt_price_per_1k_usd,
        completion_price_per_1k_usd=completion_price_per_1k_usd,
        guard_rows=guard_rows,
    )
    guard_metrics_path = None
    if guard_rows:
        guard_metrics_path = reports_dir / GUARD_METRICS_FILENAME
        write_csv(guard_metrics_path, guard_rows, list(guard_rows[0]))
    return {
        "qa_metrics_path": qa_metrics_path,
        "cost_latency_path": cost_latency_path,
        "summary_path": summary_path,
        "audit_path": audit_path,
        "guard_metrics_path": guard_metrics_path,
    }


def _parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / float(len(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = float(ordered[lower_index])
    upper_value = float(ordered[upper_index])
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def _round_float(value: float, digits: int) -> float:
    return round(float(value), digits)


def _is_answered(run_row: dict[str, Any]) -> bool:
    answer_text = run_row.get("answer_text")
    return isinstance(answer_text, str) and bool(answer_text.strip())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _answer_model_name(snapshot: dict[str, Any]) -> str:
    """The answer model a run used, from its config snapshot (new `llm:` block or flat keys)."""
    try:
        return resolve_llm_settings(snapshot).role("answer").model
    except ValueError:
        # Snapshots the settings module can't resolve (e.g. very old ones without llm_provider):
        # read the flat model keys directly, as this report always did.
        return str(snapshot.get("llm_answer_model") or snapshot.get("openai_answer_model") or "unknown").strip()
