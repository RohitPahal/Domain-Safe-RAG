"""Guard label set for owner review: cached rewrites with a suggested label from independent rules.

The domain-safe guard is evaluated against human labels (precision, recall, false-alarm rate). This
tool prepares them: it samples distinct cached multi-query rewrites (every rewrite the guard flags, a seeded
sample of the ones it passes, stratified by rewrite type, and every rewrite that leaks a gold
answer) and suggests a label for each from its own rules (they share only tokenisation helpers
with guard v2, ``guard.checks``, and nothing with the v1 guard being evaluated):

  flag   adds_answer         the rewrite contains a gold answer the question doesn't
         negation_change     a negation word appears or disappears
         answer_type_drift   "how many" / "how much" is lost
         drops_critical      an identifier (a token with a digit or 2+ capitals) is lost; kept if all
                             its words (any order), its joined form (U12-dependent = U12dependent) or
                             a corpus expansion of it (guard.aliases) are in the rewrite
  check  expanded            a lost identifier seems written out (Schwartz & Hearst letter matching,
                             e.g. DAT1 -> "dopamine transporter 1"); letters can match by chance
         paraphrase          ordinary content words are missing (could be a fine synonym)
         adds_entity         a critical-looking or capitalised word is added (possible knowledge
                             injection), unless it expands an abbreviation of the question
  ok     none of the above

The owner reviews every row and fills ``owner_label`` (ok / flag) and, for flag, ``owner_category``;
only reviewed rows count. ``guard_labels.csv`` doesn't show the guard's decision, so rows can be
labelled blind; it is in ``guard_labels_guard_view.csv`` (same ``row_id``). A file the owner has
started reviewing is never overwritten. No LLM calls.

Usage:
    python tools/guard_labels.py --output-dir outputs/reports
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis.evaluation.guard_metrics import answer_leaks
from rag_thesis.evaluation.retrieval import write_csv
from rag_thesis.guard.aliases import AbbreviationDictionary, find_expansion
from rag_thesis.guard.checks import TOKEN, identifiers, words
from rag_thesis.guard.checks import contains as _contains
from rag_thesis.guard.domain_safe import assess_rewrite
from rag_thesis.io import load_questions, read_jsonl
from rag_thesis.llm.cache import load_cache_records
from rag_thesis.protection import ensure_writable
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER
from rag_thesis.schemas import DocumentLine

BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
PASSED_SAMPLE = 150
SEED = 42
STOPWORDS = set(
    "a an the of in on at to for from by with and or as is are was were be been being do does did has have had "
    "which what who whom whose where when why how there their its it this that these those can could may might "
    "most more name used use known also called main primary one any many much".split()
)
NEGATIONS = {"not", "no", "never", "without", "none", "neither", "nor", "cannot", "non"}


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _kept(token: str, rewrite: str, aliases: AbbreviationDictionary) -> bool:
    """Is a critical token of the question still in the rewrite (any spelling, or expanded)?"""
    rewrite_words = words(rewrite)
    token_words = words(token)
    if set(token_words) <= set(rewrite_words) or "".join(token_words) in "".join(rewrite_words):
        return True
    return any(_contains(rewrite_words, words(long)) for long in aliases.expansions(token.strip("()[],.;:?")))


def suggest_label(question: str, rewrite: str, answers: list[str], aliases: AbbreviationDictionary) -> dict[str, str]:
    """Suggested label, category and reasons for one rewrite, from rules independent of the guard."""
    q_words, r_words = words(question), words(rewrite)
    flags, checks = [], []
    critical = identifiers(question)
    lost = [token for token in critical if not _kept(token, rewrite, aliases)]
    expanded = {token: find_expansion(token, rewrite) for token in lost}
    expansion_words = {w for long in expanded.values() if long for w in words(long)}
    leaked = answer_leaks(question, rewrite, answers)
    if leaked:
        flags.append(("adds_answer", f"adds the answer '{leaked[0]}'"))
    # A negation word inside a written-out abbreviation (SNF = Sucrose Non-Fermentable) isn't one.
    if bool(NEGATIONS & set(q_words)) != bool(NEGATIONS & (set(r_words) - expansion_words)):
        flags.append(("negation_change", "a negation appears or disappears"))
    for phrase in (["how", "many"], ["how", "much"]):
        if _contains(q_words, phrase) and not (
            _contains(r_words, phrase) or {"number", "count", "quantify", "quantity", "total"} & set(r_words)
        ):
            flags.append(("answer_type_drift", f"'{' '.join(phrase)}' is lost"))
    if any(long is None for long in expanded.values()):
        flags.append(("drops_critical", "loses " + ", ".join(t for t, long in expanded.items() if long is None)))
    if any(long is not None for long in expanded.values()):
        checks.append(("expanded", "; ".join(f"{t} written out as '{long}'?" for t, long in expanded.items() if long)))
    r_stems = {_stem(word) for word in r_words}
    r_joined = "".join(r_words)
    ordinary = [w for w in q_words if w not in STOPWORDS and not any(w in words(c) for c in critical)]
    missing = [w for w in dict.fromkeys(ordinary) if _stem(w) not in r_stems and w not in r_joined]
    if missing:
        checks.append(("paraphrase", "missing " + ", ".join(missing)))
    known = set(q_words) | expansion_words
    known |= {w for token in critical for long in aliases.expansions(token) for w in words(long)}
    added = [
        token
        for index, token in enumerate(TOKEN.findall(rewrite))
        if (token in identifiers(token) or (index > 0 and token[:1].isupper())) and not set(words(token)) <= known
    ]
    if added:
        checks.append(("adds_entity", "adds " + ", ".join(dict.fromkeys(added))))
    chosen = flags or checks
    return {
        "suggested_label": "flag" if flags else "check" if checks else "ok",
        "suggested_category": chosen[0][0] if chosen else "",
        "suggested_reasons": "; ".join(reason for _, reason in chosen),
    }


def sample_rows(
    questions: list[dict[str, Any]], rewrites: dict[str, list[str]], aliases: AbbreviationDictionary
) -> list[dict[str, Any]]:
    """Every rewrite the guard flags or that leaks an answer, plus a seeded stratified sample of the rest."""
    candidates: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for question in questions:
        answers = [str(answer["text"]) for answer in question["answers"]]
        for slot, rewrite in zip(MULTI_REWRITE_ORDER, rewrites[question["qid"]], strict=True):
            twin = seen.get((question["question"], rewrite))
            if twin is not None:
                twin["also_qids"] = " ".join(filter(None, [twin["also_qids"], question["qid"]]))
                continue
            assessment = assess_rewrite(question["question"].strip(), rewrite.strip())
            suggestion = suggest_label(question["question"], rewrite, answers, aliases)
            candidates.append(
                {
                    "qid": question["qid"],
                    "rewrite_type": slot,
                    "question": question["question"],
                    "rewrite": rewrite,
                    "gold_answers": " | ".join(dict.fromkeys(answers)),
                    "also_qids": "",
                    **suggestion,
                    "_guard_flagged": not assessment.is_safe,
                    "_guard_reasons": " ".join(assessment.reason_codes),
                    "_missing": " | ".join(assessment.missing_tokens),
                    "_added": " | ".join(assessment.added_tokens),
                }
            )
            seen[(question["question"], rewrite)] = candidates[-1]
    chosen = [row for row in candidates if row["_guard_flagged"] or row["suggested_category"] == "adds_answer"]
    rest = [row for row in candidates if row not in chosen]
    rng = random.Random(SEED)
    per_type = PASSED_SAMPLE // len(MULTI_REWRITE_ORDER)
    for index, slot in enumerate(MULTI_REWRITE_ORDER):
        pool = [row for row in rest if row["rewrite_type"] == slot]
        take = per_type + (1 if index < PASSED_SAMPLE % len(MULTI_REWRITE_ORDER) else 0)
        chosen.extend(rng.sample(pool, min(take, len(pool))))
    rng.shuffle(chosen)  # review order mixes flagged and passed rewrites
    for number, row in enumerate(chosen, start=1):
        row["row_id"] = f"g{number:03d}"
    return chosen


def _has_owner_input(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        return any((row.get("owner_label") or "").strip() for row in csv.DictReader(handle))


def summary_markdown(rows: list[dict[str, Any]]) -> str:
    """Counts, and how the suggestions line up with the guard (both unreviewed)."""
    table = Counter((row["suggested_label"], row["_guard_flagged"]) for row in rows)
    lines = [
        "# Guard label set (unreviewed suggestions)",
        "",
        f"{len(rows)} cached rewrites: {sum(r['_guard_flagged'] for r in rows)} the guard flags, "
        f"{sum(not r['_guard_flagged'] for r in rows)} it passes (seeded sample), every answer leak included; "
        "one row per distinct (question, rewrite), its twins in `also_qids`. "
        "Suggested labels come from rules independent of the guard; only owner-reviewed rows count.",
        "",
        "| suggested label | guard flags | guard passes |",
        "| --- | ---: | ---: |",
    ]
    for label in ("flag", "check", "ok"):
        lines.append(f"| {label} | {table[(label, True)]} | {table[(label, False)]} |")
    categories = Counter(row["suggested_category"] or "-" for row in rows)
    lines += ["", "Suggested categories: " + ", ".join(f"{name} {count}" for name, count in categories.most_common())]
    lines += [
        "",
        "## How to review",
        "",
        "Open `guard_labels.csv`. For every row, set `owner_label` to `ok` (meaning and critical information "
        "kept, nothing the question doesn't say added) or `flag` (the rewrite should not be used as is); for "
        "`flag`, set `owner_category` (adds_answer, adds_knowledge, negation_change, answer_type_drift, "
        "drops_critical, alters_critical, other) and add a note if useful. The guard's decision is in "
        "`guard_labels_guard_view.csv`; look at it only after labelling.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rewrite-cache", type=Path, default=BENCHMARK / "cache" / "rewrites.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--documents", type=Path, default=Path("data/full/processed/documents.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    labels_path = args.output_dir / "guard_labels.csv"
    if _has_owner_input(labels_path):
        raise SystemExit(f"{labels_path} already holds owner labels; not overwriting it.")
    questions = load_questions(args.questions)
    records = load_cache_records(args.rewrite_cache).values()
    rewrites = {str(r["qid"]): list(r["rewrites"]) for r in records if r.get("rewrite_mode") == "multi"}
    aliases = AbbreviationDictionary.from_texts(row["text"] for row in read_jsonl(args.documents, DocumentLine))
    rows = sample_rows(questions, rewrites, aliases)

    review_fields = ["row_id", "qid", "also_qids", "rewrite_type", "question", "rewrite", "gold_answers"]
    review_fields += ["suggested_label", "suggested_category", "suggested_reasons"]
    blank = {"owner_label": "", "owner_category": "", "owner_note": ""}
    write_csv(labels_path, [{**{f: row[f] for f in review_fields}, **blank} for row in rows], [*review_fields, *blank])
    guard_fields = {"_guard_flagged": "guard_flagged", "_guard_reasons": "guard_reason_codes"}
    guard_fields |= {"_missing": "guard_missing_tokens", "_added": "guard_added_tokens"}
    write_csv(
        args.output_dir / "guard_labels_guard_view.csv",
        [{"row_id": row["row_id"], **{new: row[old] for old, new in guard_fields.items()}} for row in rows],
        ["row_id", *guard_fields.values()],
    )
    summary_path = args.output_dir / "guard_labels_summary.md"
    ensure_writable(summary_path)  # never write into a protected folder (see protection.py)
    summary = summary_markdown(rows)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
