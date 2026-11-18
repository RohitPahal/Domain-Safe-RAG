"""Corpus abbreviation dictionary: short forms and their long forms, found with Schwartz & Hearst.

Schwartz & Hearst (2003), "A simple algorithm for identifying abbreviation definitions in
biomedical text": a definition is written ``long form (SF)``. The short form is the text in
brackets (2-10 characters, at most two words, starting with a letter or digit, containing a
letter); the long form is the shortest run of words just before the bracket in which every
letter or digit of the short form appears in order, the first one at the start of a word.

``AbbreviationDictionary.from_texts`` counts every definition in a corpus, so a short form maps to
its long forms, most frequent first. Used to tell an expanded abbreviation from a lost one
(guard), and to expand abbreviations in a query without an LLM (retrieval).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

_DEFINITION = re.compile(r"\(([^()]{2,10})\)")
_WORD = re.compile(r"\S+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./-]*[A-Za-z0-9]|[A-Za-z0-9]")


def extract_abbreviations(text: str) -> list[tuple[str, str]]:
    """Every ``(short form, long form)`` definition in ``text``, in order."""
    found = []
    for match in _DEFINITION.finditer(text):
        short = match.group(1).strip()
        if not _valid_short_form(short):
            continue
        words = _WORD.findall(text[: match.start()])
        window = words[-min(len(short) + 5, 2 * len(short)) :]
        long = _best_long_form(short, " ".join(window))
        if long and len(long) > len(short) and short.lower() not in long.lower().split():
            found.append((short, long))
    return found


def find_expansion(short: str, text: str) -> str | None:
    """A long form of ``short`` written out anywhere in ``text``, found with the same matching at
    every word end (e.g. "dopamine transporter 1" for DAT1), or ``None``. Letters can match by
    chance, so treat a hit as a candidate, not a proof."""
    if not 2 <= len(short) <= 10 or not _valid_short_form(short):
        return None
    words = _WORD.findall(text)
    span = min(len(short) + 5, 2 * len(short))
    for end in range(1, len(words) + 1):
        long = _best_long_form(short, " ".join(words[max(0, end - span) : end]))
        if long and len(long) > len(short) and short.lower() not in long.lower().split():
            return long
    return None


def _valid_short_form(short: str) -> bool:
    return (
        len(short.split()) <= 2
        and short[0].isalnum()
        and any(character.isalpha() for character in short)
        and not short.isdigit()
    )


def _best_long_form(short: str, candidate: str) -> str | None:
    """The shortest end of ``candidate`` that holds the short form's characters in order (S&H)."""
    s_index, l_index = len(short) - 1, len(candidate) - 1
    while s_index >= 0:
        character = short[s_index].lower()
        if not character.isalnum():
            s_index -= 1
            continue
        while l_index >= 0 and (
            candidate[l_index].lower() != character
            or (s_index == 0 and l_index > 0 and candidate[l_index - 1].isalnum())
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    start = candidate.rfind(" ", 0, l_index + 1) + 1
    long = candidate[start:].strip(" ,;:")
    return long or None


class AbbreviationDictionary:
    """Short form -> long forms seen in a corpus (lower-cased, most frequent first)."""

    def __init__(self, counts: dict[str, Counter[str]]) -> None:
        self._counts = counts

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> AbbreviationDictionary:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for text in texts:
            for short, long in extract_abbreviations(text):
                counts[short.lower()][" ".join(long.lower().split())] += 1
        return cls(dict(counts))

    def expansions(self, short: str) -> list[str]:
        """Long forms of ``short`` (any case), most frequent first; ties alphabetically."""
        counts = self._counts.get(short.lower(), Counter())
        return [long for long, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

    def expand(self, text: str) -> str:
        """``text`` with the most frequent long form added in brackets after each identifier-like
        token (a digit or two capitals) the dictionary knows: "Where is DAT1?" -> "Where is DAT1
        (dopamine transporter 1)?". The short form stays, so nothing is lost; a long form already
        in the text isn't added again."""

        def add_long_form(match: re.Match[str]) -> str:
            token = match.group(0)
            if not (any(c.isdigit() for c in token) or sum(c.isupper() for c in token) >= 2):
                return token
            long_forms = self.expansions(token)
            if not long_forms or long_forms[0] in " ".join(text.lower().split()):
                return token
            return f"{token} ({long_forms[0]})"

        return _TOKEN.sub(add_long_form, text)

    def __len__(self) -> int:
        return len(self._counts)
