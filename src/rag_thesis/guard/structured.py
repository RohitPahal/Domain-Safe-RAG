"""R6 guard: check each rewrite against the structure the rewriter stated for the question.

The ``structured`` rewrite mode returns, with the four rewrites, the question's entities,
abbreviations, constraints and answer type (``rewrite.rewriter.StructuredRewrite``). The model
wrote that structure, so it is checked first: an entity, constraint or abbreviation that isn't in
the question is invented and isn't trusted (a warning, and its words don't count as known). Then
every rewrite must keep each real entity (or its stated or corpus long form) and the content words
of each constraint, keep the answer type and the negation (``guard.checks``), and add no code
(block) or capitalised name (``names``) beyond the question and the long forms. Offline assessor,
like ``guard.checks``: evaluated before the pipeline uses it.
"""

from __future__ import annotations

from rag_thesis.guard.aliases import AbbreviationDictionary
from rag_thesis.guard.checks import (
    BLOCK,
    TOKEN,
    WARN,
    Violation,
    check_answer_type,
    check_negation,
    contains,
    identifiers,
    words,
)
from rag_thesis.rewrite.rewriter import StructuredRewrite

_FUNCTION_WORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "is", "are"}


def _in_text(phrase: str, text_words: list[str]) -> bool:
    phrase_words = words(phrase)
    return contains(text_words, phrase_words) or (bool(phrase_words) and "".join(phrase_words) in "".join(text_words))


def check_structure(question: str, structure: StructuredRewrite) -> list[Violation]:
    """Entities, constraints and short forms the structure states but the question doesn't contain."""
    question_words = words(question)
    stated = [("entity", e) for e in structure.entities] + [("constraint", c) for c in structure.constraints]
    stated += [("abbreviation", a.short) for a in structure.abbreviations]
    return [
        Violation("structure", WARN, f"{kind} '{text}' is not in the question")
        for kind, text in stated
        if not _in_text(text, question_words)
    ]


def check_rewrite(
    question: str,
    structure: StructuredRewrite,
    rewrite: str,
    aliases: AbbreviationDictionary | None = None,
    *,
    names: str = WARN,
) -> list[Violation]:
    """One rewrite against the (trusted part of the) structure."""
    aliases = aliases or AbbreviationDictionary({})
    question_words, rewrite_words = words(question), words(rewrite)
    long_forms = {a.short.lower(): a.long for a in structure.abbreviations if _in_text(a.short, question_words)}
    out: list[Violation] = []
    for entity in structure.entities:
        if not _in_text(entity, question_words):
            continue  # invented by the model: reported by check_structure, not required
        expansions = [long_forms[entity.lower()]] if entity.lower() in long_forms else []
        expansions += aliases.expansions(entity)
        if not (_in_text(entity, rewrite_words) or any(_in_text(long, rewrite_words) for long in expansions)):
            out.append(Violation("entities", BLOCK, f"loses {entity}"))
    for constraint in structure.constraints:
        if not _in_text(constraint, question_words):
            continue
        content = [w for w in words(constraint) if w not in _FUNCTION_WORDS]
        if not set(content) <= set(rewrite_words):
            out.append(Violation("constraints", BLOCK, f"loses the constraint '{constraint}'"))
    out += check_answer_type(question, rewrite, aliases) + check_negation(question, rewrite, aliases)
    known = set(question_words) | {w for long in long_forms.values() for w in words(long)}
    known |= {w for token in identifiers(question) for long in aliases.expansions(token) for w in words(long)}
    seen: set[str] = set()
    for index, token in enumerate(TOKEN.findall(rewrite)):
        if set(words(token)) <= known or token in seen:
            continue
        seen.add(token)
        if token in identifiers(token):
            out.append(Violation("added_entities", BLOCK, f"adds {token}"))
        elif index > 0 and token[:1].isupper():
            out.append(Violation("added_entities", names, f"adds {token}"))
    return out


def assess_structured(
    question: str,
    structure: StructuredRewrite,
    aliases: AbbreviationDictionary | None = None,
    *,
    names: str = WARN,
) -> tuple[list[Violation], list[list[Violation]]]:
    """The structure's own violations, and each of the four rewrites' violations (in type order)."""
    if names not in (BLOCK, WARN):
        raise ValueError(f"names must be '{BLOCK}' or '{WARN}', not {names!r}.")
    per_rewrite = [
        check_rewrite(question, structure, rewrite, aliases, names=names) for rewrite in structure.rewrite_list()
    ]
    return check_structure(question, structure), per_rewrite
