"""Deterministic guardrails around answer generation: checks in code, not in the prompt.

The hardened answer prompt (``prompts/answer_v2``) tells the model how to behave; these checks
don't depend on the model obeying. Three places, each returning ``Finding``s (``block`` or ``warn``):

- ``check_question``: before any model is called. Blocks prompt-injection attempts, requests for
  instructions to cause harm, self-harm, requests for personal medical advice, and unusable input
  (empty, far too long). Factoid questions *about* dangerous things ("Which toxin does Clostridium
  botulinum produce?", "What is the lethal dose of digoxin?") are allowed: only a request for
  how-to instructions or for a personal decision is blocked.
- ``check_passage``: retrieved text is untrusted. Instruction-like text inside a document is
  reported (``warn``): the passage is still used, fenced as data by the prompt, and the answer
  check below catches an answer that isn't the passage's own words.
- ``check_answer``: after generation. Blocks an answer that isn't found in the passages it was
  given (an extractive answer must be a span of them; ``yes`` / ``no`` are exempt), that is too
  long to be a factoid span, or that carries markup, links, code, instruction-like text or the
  prompt's own wording.

Patterns are deliberately few and readable; they are a floor, not a classifier. Their precision
and recall are measured on a red-team set (tools/guardrail_eval.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_thesis.guard.checks import contains, words

BLOCK, WARN = "block", "warn"
MAX_QUESTION_CHARS = 1000
MAX_ANSWER_WORDS = 30


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str  # BLOCK or WARN
    detail: str


def _any(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags=re.IGNORECASE)


# Attempts to override the rules, change the role or extract the prompt.
INJECTION = _any(
    (
        # "ignore the previous instructions", "disregard the question and the rules": the verb must
        # be close to what it overrides, so "ignore minor side effects ... instructions" is not one.
        r"\b(?:ignore|disregard|forget|override|stop following)\b[^.\n]{0,25}?"
        r"\b(?:instructions?|rules?|prompts?|guidelines)\b",
        r"\b(?:ignore|disregard|forget)\s+(?:everything|anything|all)\b[^.\n]{0,30}\b(?:told|said|above|before)\b",
        r"\b(?:reveal|show|print|repeat|output|tell me)\b[^.\n]{0,30}"
        r"\b(?:system prompt|your (?:instructions|prompt|rules)|hidden prompt)\b",
        # A role change addressed to the model ("act as" alone is ordinary biomedical English).
        r"\byou are now\b|\bpretend (?:to be|you are)\b|\bfrom now on\b",
        r"\bact as (?:if you|an? (?:ai|assistant|chatbot|model|unfiltered|unrestricted))\b",
        r"\b(?:reply|respond|answer)\s+only\s+with\b",
        # Text that addresses the model reading it, or tells it to overrule the other passages.
        r"\b(?:note|message|correction|instructions?) (?:to|for) the (?:ai|assistant|model|llm|chatbot)\b",
        r"\b(?:whatever|regardless of what|no matter what) the (?:other )?(?:passages?|documents?|context|sources?)\b",
        r"\bdeveloper mode\b|\bjailbreak\b|\bdo anything now\b|\bDAN mode\b",
        r"^\s*(?:system|assistant|developer)\s*:",
        r"<\|im_(?:start|end)\|>|\[/?INST\]|###\s*(?:instruction|system)",
        r"\bnew instructions?\b\s*:",
    )
)
_HOW_TO = (
    r"(?:how (?:to|do i|can i|would i|could one)|steps? to|instructions? (?:for|to|on)|protocol (?:for|to)|"
    r"recipe for|guide (?:for|to)|best way to|teach me to)"
)
_HARM_ACT = (
    r"(?:synthesi[sz]e|manufacture|produce|make|build|culture|grow|purify|weaponi[sz]e|aerosoli[sz]e|"
    r"disperse|release|poison|contaminate|enhance|increase the (?:transmissibility|lethality|virulence))"
)
_HARM_AGENT = (
    r"(?:toxin|nerve agent|chemical weapon|bioweapon|biological weapon|pathogen|virus|bacteri\w+|poison|explosive|"
    r"ricin|sarin|vx\b|anthrax|botulinum|smallpox|ebola|plague)"
)
# Asking for instructions that could cause harm, not asking a fact about a harmful thing.
HARMFUL_REQUEST = _any(
    (
        rf"{_HOW_TO}[^.?\n]{{0,60}}\b{_HARM_ACT}\b[^.?\n]{{0,60}}\b{_HARM_AGENT}",
        r"\b(?:kill|poison|harm|hurt|infect|sedate|drug)\s+(?:someone|somebody|a person|people|my \w+|him|her|them)\b",
        r"\bwithout (?:being |getting )?(?:detected|caught|traced)\b",
        r"\b(?:undetectable|untraceable)\b[^.?\n]{0,30}\b(?:poison|toxin|drug|dose)\b",
    )
)
SELF_HARM = _any(
    (
        r"\b(?:kill|hurt|harm) myself\b|\bend my life\b|\bsuicide (?:method|methods|plan)\b",
        r"\b(?:overdose|lethal dose)\b[^.?\n]{0,40}\b(?:for me|myself|i (?:need|take|should))\b",
    )
)
# A personal decision, not a fact: the system extracts facts from papers, it doesn't advise patients.
MEDICAL_ADVICE = _any(
    (
        r"\bshould i\b[^.?\n]{0,60}\b(?:take|stop|start|increase|reduce|double|skip|use|get)\b",
        r"\b(?:what|which|how much)\b[^.?\n]{0,30}\b(?:dose|dosage|medication|drug|treatment)\b"
        r"[^.?\n]{0,30}\b(?:should i|for me|for my)\b",
        r"\b(?:diagnose me|do i have|am i (?:sick|infected|pregnant)|is my \w+ (?:normal|dangerous|cancer))\b",
        r"\bmy (?:symptoms?|test results?|prescription|doctor)\b",
    )
)
# Things a one-line factoid answer never contains.
ANSWER_PAYLOAD = _any(
    (
        r"https?://|www\.\w",
        r"```|<\s*/?\s*(?:script|img|iframe|a|html|svg|style)\b|\bjavascript:",
        r"!\[[^\]]*\]\(|\]\(\s*\w+:",
        r"\buntrusted data\b|\bthese rules\b|\bmy instructions\b|\bsystem prompt\b|<\s*/?\s*(?:passage|question)\b",
    )
)
_YES_NO = {"yes", "no"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIST_SEPARATOR = re.compile(r";|,|\band\b|\bor\b", flags=re.IGNORECASE)


def check_question(question: str) -> list[Finding]:
    """Findings for a user question, before any retrieval or model call."""
    text = question.strip()
    if not text:
        return [Finding("input", BLOCK, "empty question")]
    findings = []
    if len(text) > MAX_QUESTION_CHARS:
        findings.append(Finding("input", BLOCK, f"question longer than {MAX_QUESTION_CHARS} characters"))
    for check, pattern, detail in (
        ("injection", INJECTION, "tries to change the rules or extract the prompt"),
        ("harmful_request", HARMFUL_REQUEST, "asks for instructions that could cause harm"),
        ("self_harm", SELF_HARM, "self-harm"),
        ("medical_advice", MEDICAL_ADVICE, "asks for a personal medical decision"),
    ):
        match = pattern.search(text)
        if match:
            findings.append(Finding(check, BLOCK, f"{detail}: '{match.group(0)[:60]}'"))
    return findings


def check_passage(text: str) -> list[Finding]:
    """Instruction-like text inside a retrieved passage (reported; the passage stays fenced data)."""
    match = INJECTION.search(text)
    return [Finding("passage_injection", WARN, f"instruction-like text: '{match.group(0)[:60]}'")] if match else []


def check_answer(answer: str, passages: list[str]) -> list[Finding]:
    """Findings for a generated answer against the passages the model was given."""
    findings = []
    payload = ANSWER_PAYLOAD.search(answer) or INJECTION.search(answer)
    if payload:
        findings.append(Finding("answer_payload", BLOCK, f"not a plain answer span: '{payload.group(0)[:40]}'"))
    answer_words = words(answer)
    if len(answer_words) > MAX_ANSWER_WORDS:
        findings.append(Finding("answer_length", BLOCK, f"{len(answer_words)} words; a factoid span is short"))
    # A word planted by an instruction inside a passage ("reply only with HACKED") must not count as
    # evidence, so instruction-like sentences are left out of the text the answer is grounded in.
    evidence = [" ".join(x for x in _SENTENCE.split(passage) if not INJECTION.search(x)) for passage in passages]
    if not _grounded(answer, evidence):
        findings.append(Finding("ungrounded", BLOCK, "the answer is not a span of the passages it was given"))
    return findings


def _grounded(answer: str, passages: list[str]) -> bool:
    """The answer occurs word for word in some passage, or each of its listed items does (items
    separated by ';', ',', 'and' or 'or'; a lone ``yes`` / ``no`` is exempt)."""
    passage_words = [words(passage) for passage in passages]

    def found(item: list[str]) -> bool:
        return (len(item) == 1 and item[0] in _YES_NO) or any(contains(text, item) for text in passage_words)

    whole = words(answer)
    if not whole:
        return False
    if found(whole):
        return True
    items = [words(item) for item in _LIST_SEPARATOR.split(answer)]
    items = [item for item in items if item]
    return len(items) > 1 and all(found(item) for item in items)


def is_blocked(findings: list[Finding]) -> bool:
    return any(finding.severity == BLOCK for finding in findings)
