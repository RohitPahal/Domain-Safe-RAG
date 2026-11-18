"""Guard evaluation: v1 and v2 against the owner's labels, and their decisions on every cached rewrite.

Reads the owner-reviewed rows of ``guard_labels.csv`` (``owner_label`` ok / flag; see
tools/guard_labels.py) and scores three guards on them: v1 (``guard.domain_safe.assess_rewrite``)
and v2 (``guard.checks``) with added capitalised names warning or blocking. A guard *flags* a
rewrite when it would repair or drop it. Per guard: precision (flags that are real), recall (real
problems flagged) and false-alarm rate (fine rewrites flagged), each with a 95% bootstrap CI that
resamples BioASQ questions; v2 against v1 with exact sign tests on the rows where they disagree.

The label set holds every rewrite v1 flags but only a sample of those it passes, so the rates are
rates on the labelled set: right for comparing guards on the same rows, not population rates.

Also writes every distinct cached rewrite with each guard's decision (no labels needed).
No LLM calls. Writes guard_eval.md and guard_eval_decisions.csv into ``--output-dir``.

Usage:
    python tools/guard_eval.py --output-dir outputs/reports
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis.evaluation.retrieval import bioasq_question_id, write_csv
from rag_thesis.evaluation.stats import bootstrap_ci, sign_test
from rag_thesis.guard.aliases import AbbreviationDictionary
from rag_thesis.guard.checks import BLOCK, WARN, assess, is_blocked
from rag_thesis.guard.domain_safe import assess_rewrite
from rag_thesis.io import read_jsonl
from rag_thesis.llm.cache import load_cache_records
from rag_thesis.protection import ensure_writable
from rag_thesis.schemas import DocumentLine

BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
GUARDS = ("v1", "v2_names_warn", "v2_names_block")


def decisions(question: str, rewrite: str, aliases: AbbreviationDictionary) -> dict[str, Any]:
    """Each guard's decision (True = flags the rewrite) and v2's violations."""
    question, rewrite = question.strip(), rewrite.strip()
    v2 = assess(question, rewrite, aliases, names=WARN)
    return {
        "v1": not assess_rewrite(question, rewrite).is_safe,
        "v2_names_warn": is_blocked(v2),
        "v2_names_block": is_blocked(assess(question, rewrite, aliases, names=BLOCK)),
        "v2_violations": "; ".join(f"{v.severity}: {v.check} ({v.detail})" for v in v2),
    }


def reviewed_rows(labels_path: Path) -> list[dict[str, str]]:
    """Rows of the label file the owner has reviewed (``owner_label`` ok or flag)."""
    if not labels_path.is_file():
        return []
    with labels_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["row_id"]: (row.get("owner_label") or "").strip().lower() for row in rows}
    unknown = sorted({label for label in labels.values() if label not in ("", "ok", "flag")})
    if unknown:
        raise ValueError(f"owner_label must be 'ok' or 'flag', found {unknown} in {labels_path}.")
    return [row for row in rows if labels[row["row_id"]]]


def scores(rows: list[dict[str, Any]], guard: str) -> dict[str, Any]:
    """Confusion counts and precision / recall / false-alarm rate (with CIs) of one guard."""
    real = [row["owner_label"].strip().lower() == "flag" for row in rows]
    flagged = [bool(row[guard]) for row in rows]
    clusters = [bioasq_question_id(row["qid"]) for row in rows]

    def rate(selected: list[int], value: list[bool]) -> tuple[float, float, float] | None:
        if not selected:
            return None
        return bootstrap_ci([float(value[i]) for i in selected], clusters=[clusters[i] for i in selected])

    counts = Counter(zip(flagged, real, strict=True))
    return {
        "guard": guard,
        "tp": counts[(True, True)],
        "fp": counts[(True, False)],
        "fn": counts[(False, True)],
        "tn": counts[(False, False)],
        "precision": rate([i for i, f in enumerate(flagged) if f], real),
        "recall": rate([i for i, r in enumerate(real) if r], flagged),
        "false_alarm_rate": rate([i for i, r in enumerate(real) if not r], flagged),
    }


def _ci(values: tuple[float, float, float] | None) -> str:
    return "n/a" if values is None else f"{values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"


def report_markdown(all_rows: list[dict[str, Any]], labelled: list[dict[str, Any]]) -> str:
    lines = [
        "# Guard evaluation (post-thesis result, not a thesis number)",
        "",
        f"## Decisions on all {len(all_rows)} distinct cached rewrites (no labels needed)",
        "",
        "| guard | flags |",
        "| --- | ---: |",
        *[f"| {guard} | {sum(bool(row[guard]) for row in all_rows)} |" for guard in GUARDS],
        "",
        f"v1 flags, v2 (names warn) passes: {sum(r['v1'] and not r['v2_names_warn'] for r in all_rows)}; "
        f"v1 passes, v2 (names warn) flags: {sum(r['v2_names_warn'] and not r['v1'] for r in all_rows)}.",
        "",
        "## Against the owner's labels",
        "",
    ]
    if not labelled:
        lines.append("No reviewed rows yet: fill `owner_label` in `guard_labels.csv` (tools/guard_labels.py).")
        return "\n".join(lines) + "\n"
    real = sum(row["owner_label"].strip().lower() == "flag" for row in labelled)
    lines += [
        f"{len(labelled)} reviewed rows ({real} flag, {len(labelled) - real} ok). Rates on the labelled set, which "
        "holds every rewrite v1 flags but only a sample of the others: for comparing guards, not population rates.",
        "",
        "| guard | TP | FP | FN | TN | precision | recall | false-alarm rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for guard in GUARDS:
        s = scores(labelled, guard)
        lines.append(
            f"| {guard} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} | {_ci(s['precision'])} | "
            f"{_ci(s['recall'])} | {_ci(s['false_alarm_rate'])} |"
        )
    lines += ["", "v2 against v1 on the rows where they disagree (exact sign test):", ""]
    for guard in GUARDS[1:]:
        ok_rows = [r for r in labelled if r["owner_label"].strip().lower() == "ok"]
        flag_rows = [r for r in labelled if r["owner_label"].strip().lower() == "flag"]
        fewer = sum(r["v1"] and not r[guard] for r in ok_rows)
        more = sum(r[guard] and not r["v1"] for r in ok_rows)
        caught = sum(r[guard] and not r["v1"] for r in flag_rows)
        missed = sum(r["v1"] and not r[guard] for r in flag_rows)
        lines.append(
            f"- {guard}: false alarms removed {fewer}, added {more} (p={sign_test(fewer, more):.3g}); "
            f"real problems newly caught {caught}, newly missed {missed} (p={sign_test(caught, missed):.3g})."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=Path("outputs/reports/guard_labels.csv"))
    parser.add_argument("--rewrite-cache", type=Path, default=BENCHMARK / "cache" / "rewrites.jsonl")
    parser.add_argument("--documents", type=Path, default=Path("data/full/processed/documents.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    aliases = AbbreviationDictionary.from_texts(row["text"] for row in read_jsonl(args.documents, DocumentLine))
    pairs: dict[tuple[str, str], str] = {}
    for record in load_cache_records(args.rewrite_cache).values():
        for rewrite in record.get("rewrites", []):
            pairs.setdefault((str(record["query_original"]).strip(), rewrite.strip()), str(record["qid"]))
    all_rows = [{"qid": qid, "question": q, "rewrite": r, **decisions(q, r, aliases)} for (q, r), qid in pairs.items()]
    labelled = [{**row, **decisions(row["question"], row["rewrite"], aliases)} for row in reviewed_rows(args.labels)]

    write_csv(args.output_dir / "guard_eval_decisions.csv", all_rows, list(all_rows[0]))
    report_path = args.output_dir / "guard_eval.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    report = report_markdown(all_rows, labelled)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
