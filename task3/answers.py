"""Answer extraction, normalization, equivalence grouping and grading.

MATH: math-verify equivalence (answers wrapped as inline LaTeX). MathVista: the released testmini metadata with
MathVista's answer-type, choice and precision rules (calculate_score.py); multiple-choice answers become option letters.
Groups are formed without gold answers; the gold answer is then graded with the same equivalence rule.
"""

import platform
import re

from task3.config import NO_ANSWER

USE_TIMEOUTS = platform.system() != "Windows"


def last_boxed(text):
    """Content of the last literal \\boxed{ with balanced braces; None if absent, empty or unclosed."""
    if text is None:
        return None
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    index = start + len("\\boxed{")
    depth = 1
    for position in range(index, len(text)):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                content = text[index:position].strip()
                return content or None
    return None


def scorer_versions():
    import importlib.metadata as metadata

    versions = {}
    for package in ("math-verify", "latex2sympy2_extended", "sympy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["mathvista_rules"] = "calculate_score.py normalize_extracted_answer (answer_type, choices, precision)"
    return versions


class MathEquivalence:
    """Caches parses; two answers are equivalent when math-verify verifies them (either direction)."""

    def __init__(self):
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify
        self._parsed = {}
        self._kwargs_parse = {} if USE_TIMEOUTS else {"parsing_timeout": None}
        self._kwargs_verify = {} if USE_TIMEOUTS else {"timeout_seconds": None}

    def parsed(self, answer):
        if answer not in self._parsed:
            try:
                self._parsed[answer] = self._parse(f"${answer}$", **self._kwargs_parse)
            except Exception:
                self._parsed[answer] = []
        return self._parsed[answer]

    def equivalent(self, first, second):
        if first == second:
            return True
        left, right = self.parsed(first), self.parsed(second)
        if not left or not right:
            return False
        try:
            return bool(self._verify(left, right, **self._kwargs_verify)
                        or self._verify(right, left, **self._kwargs_verify))
        except Exception:
            return False


def group_math_answers(answers, equivalence):
    """Assign each raw answer to the first earlier group whose representative is equivalent."""
    representatives = []
    labels = []
    for answer in answers:
        if answer is None or not str(answer).strip():
            labels.append(NO_ANSWER)
            continue
        answer = str(answer).strip()
        for representative in representatives:
            if equivalence.equivalent(representative, answer):
                labels.append(representative)
                break
        else:
            representatives.append(answer)
            labels.append(answer)
    return labels


def grade_math(groups, gold, equivalence):
    gold = last_boxed(gold) or str(gold).strip()
    return {group: group != NO_ANSWER and equivalence.equivalent(group, gold) for group in set(groups)}


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, 1):
        current = [index]
        for position, right in enumerate(second, 1):
            current.append(min(previous[position] + 1, current[position - 1] + 1,
                               previous[position - 1] + (left != right)))
        previous = current
    return previous[-1]


def most_similar(prediction, choices):
    distances = [edit_distance(prediction, choice) for choice in choices]
    return choices[distances.index(min(distances))]


def letter(index):
    return chr(ord("A") + index)


def normalize_mathvista(extraction, meta):
    """MathVista's normalize_extracted_answer; multiple-choice results are returned as option letters."""
    if extraction is None or (isinstance(extraction, str) and not extraction.strip()):
        return NO_ANSWER
    question_type = meta["question_type"]
    answer_type = meta["answer_type"]
    if question_type == "multi_choice":
        choices = [str(choice) for choice in meta["choices"]]
        text = str(extraction).strip()
        bracketed = re.findall(r"\(([a-zA-Z])\)", text)
        if bracketed:
            text = bracketed[0].upper()
        options = [letter(index) for index in range(len(choices))]
        if text in options:
            return text
        if text.upper() in options and len(text) == 1:
            return text.upper()
        if text in choices:
            return letter(choices.index(text))
        return letter(choices.index(most_similar(text, choices)))
    try:
        if answer_type == "integer":
            return str(int(float(str(extraction).replace(",", "").strip())))
        if answer_type == "float":
            return str(round(float(str(extraction).replace(",", "").strip()), int(meta["precision"])))
        if answer_type == "list":
            return str(extraction)
    except (ValueError, TypeError, OverflowError):
        return NO_ANSWER
    return str(extraction).strip()


def mathvista_gold(meta):
    if meta["question_type"] == "multi_choice":
        choices = [str(choice) for choice in meta["choices"]]
        answer = str(meta["answer"])
        return letter(choices.index(answer if answer in choices else most_similar(answer, choices)))
    return normalize_mathvista(meta["answer"], meta)


def group_mathvista_answers(answers, meta):
    return [normalize_mathvista(answer, meta) for answer in answers]


def grade_mathvista(groups, meta):
    gold = mathvista_gold(meta)
    return {group: group != NO_ANSWER and group == gold for group in set(groups)}
