"""Guard v2: what a rewrite must keep and must not add, as small independent checks.

Each check takes the question, a rewrite and the corpus abbreviation dictionary and returns its
violations; ``assess`` runs the selected checks. A violation's severity is ``block`` (the rewrite
must not be used as is) or ``warn`` (recorded only). Compared with v1 (``domain_safe.assess_rewrite``):

- tokens are compared after normalisation (case, Unicode, hyphens, brackets): "U12-dependent" and
  "U12 dependent" are the same token (G1);
- an abbreviation counts as kept when its corpus long form is in the rewrite, and a long form the
  rewrite writes out itself only warns (G2): "DNA" -> "deoxyribonucleic acid" is no longer a false alarm;
- negation is checked in both directions, and losing "how many" / "how much" blocks (G3);
- added codes block, and added capitalised names (knowledge the question doesn't contain, e.g. an
  answer the rewriter knew) have a configurable severity (G4).

This module only assesses; it is evaluated offline on owner-labelled rewrites before the pipeline
uses it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from rag_thesis.guard.aliases import AbbreviationDictionary, find_expansion

BLOCK, WARN = "block", "warn"
NEGATIONS = {"not", "no", "never", "without", "none", "neither", "nor", "cannot", "non"}
COUNT_WORDS = {"number", "count", "quantify", "quantity", "total"}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./-]*[A-Za-z0-9]|[A-Za-z0-9]")


@dataclass(frozen=True)
class Violation:
    check: str
    severity: str  # BLOCK or WARN
    detail: str


def words(text: str) -> list[str]:
    """NFKC, lower case, letters and digits only (hyphens, slashes and brackets become breaks)."""
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def contains(haystack: list[str], needle: list[str]) -> bool:
    """Is ``needle`` a consecutive run of ``haystack``?"""
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def identifiers(text: str) -> list[str]:
    """Tokens with a digit or 2+ capitals (gene names, codes, abbreviations), split at brackets."""
    return [t for t in TOKEN.findall(text) if any(c.isdigit() for c in t) or sum(c.isupper() for c in t) >= 2]


def _kept(token: str, rewrite_words: list[str], aliases: AbbreviationDictionary) -> bool:
    token_words = words(token)
    if set(token_words) <= set(rewrite_words) or "".join(token_words) in "".join(rewrite_words):
        return True
    return any(contains(rewrite_words, words(long)) for long in aliases.expansions(token))


def _written_out(question: str, rewrite: str, aliases: AbbreviationDictionary) -> dict[str, str]:
    """Question identifiers the rewrite lost as written but seems to spell out: token -> long form."""
    rewrite_words = words(rewrite)
    lost = [token for token in identifiers(question) if not _kept(token, rewrite_words, aliases)]
    return {token: long for token in lost if (long := find_expansion(token, rewrite))}


def _abbreviates(token: str, question: str, aliases: AbbreviationDictionary) -> bool:
    """Is ``token`` an abbreviation of words the question already has?"""
    question_words = set(words(question))
    return find_expansion(token, question) is not None or any(
        set(words(long)) <= question_words for long in aliases.expansions(token)
    )


def check_identifiers(question: str, rewrite: str, aliases: AbbreviationDictionary, **_: str) -> list[Violation]:
    """Every identifier of the question must survive (any spelling, or its corpus long form)."""
    rewrite_words = words(rewrite)
    out = []
    for token in dict.fromkeys(identifiers(question)):
        if _kept(token, rewrite_words, aliases):
            continue
        long = find_expansion(token, rewrite)
        if long:
            out.append(Violation("identifiers", WARN, f"{token} written out as '{long}'?"))
        else:
            out.append(Violation("identifiers", BLOCK, f"loses {token}"))
    return out


def check_negation(question: str, rewrite: str, aliases: AbbreviationDictionary, **_: str) -> list[Violation]:
    """A negation may neither appear nor disappear (one inside a written-out abbreviation, as in
    SNF = Sucrose Non-Fermentable, isn't a negation)."""
    spelled = {w for long in _written_out(question, rewrite, aliases).values() for w in words(long)}
    if bool(NEGATIONS & set(words(question))) != bool(NEGATIONS & (set(words(rewrite)) - spelled)):
        return [Violation("negation", BLOCK, "a negation appears or disappears")]
    return []


def check_answer_type(question: str, rewrite: str, aliases: AbbreviationDictionary, **_: str) -> list[Violation]:
    """A count question must stay a count question."""
    q_words, r_words = words(question), words(rewrite)
    for phrase in (["how", "many"], ["how", "much"]):
        if contains(q_words, phrase) and not (contains(r_words, phrase) or COUNT_WORDS & set(r_words)):
            return [Violation("answer_type", BLOCK, f"'{' '.join(phrase)}' is lost")]
    return []


def check_added_entities(
    question: str, rewrite: str, aliases: AbbreviationDictionary, *, names: str = WARN, **_: str
) -> list[Violation]:
    """Codes and capitalised names the question doesn't contain (knowledge injection). Words of a
    long form of one of the question's abbreviations don't count, nor does a code that abbreviates
    words of the question ("FIPA" for "familial isolated pituitary adenoma"). Added codes block;
    added names get ``names`` as their severity."""
    known = set(words(question))
    known |= {w for token in identifiers(question) for long in aliases.expansions(token) for w in words(long)}
    known |= {w for long in _written_out(question, rewrite, aliases).values() for w in words(long)}
    out, seen = [], set()
    for index, token in enumerate(TOKEN.findall(rewrite)):
        if set(words(token)) <= known or token in seen:
            continue
        seen.add(token)
        if token in identifiers(token):
            if not _abbreviates(token, question, aliases):
                out.append(Violation("added_entities", BLOCK, f"adds {token}"))
        elif index > 0 and token[:1].isupper():
            out.append(Violation("added_entities", names, f"adds {token}"))
    return out


CHECKS: dict[str, Callable[..., list[Violation]]] = {
    "identifiers": check_identifiers,
    "negation": check_negation,
    "answer_type": check_answer_type,
    "added_entities": check_added_entities,
}


def assess(
    question: str,
    rewrite: str,
    aliases: AbbreviationDictionary,
    *,
    checks: tuple[str, ...] = tuple(CHECKS),
    names: str = WARN,
) -> list[Violation]:
    """All violations of ``rewrite`` under the selected ``checks``; ``names`` is the severity of an
    added capitalised name (``warn`` or ``block``)."""
    if names not in (BLOCK, WARN):
        raise ValueError(f"names must be '{BLOCK}' or '{WARN}', not {names!r}.")
    return [violation for name in checks for violation in CHECKS[name](question, rewrite, aliases, names=names)]


def is_blocked(violations: list[Violation]) -> bool:
    """Must the rewrite be repaired or dropped?"""
    return any(violation.severity == BLOCK for violation in violations)
