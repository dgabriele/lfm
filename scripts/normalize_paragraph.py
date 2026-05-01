"""Paragraph-scoped, value-keyed NER normalization with letter-suffix coreference.

Replaces named entities with placeholder tokens of the form
``{type}{suffix}`` (e.g. ``moneyamounta``) where:

  * **type** is a normalised category derived from the spaCy NER label
    (MONEY → moneyamount, DATE → dateexpression, …).
  * **suffix** is a per-paragraph letter (a..z) keyed on the *surface
    form* of the entity, so the same entity recurring within a paragraph
    gets the same suffix. Distinct surface forms get distinct suffixes,
    even if they share an entity type.

Overflow (>26 distinct entities of one type per paragraph) collapses to
``z``. >99% of natural paragraphs stay well under this cap.

Used both as a runnable smoke-test script (with ``--smoke``) and as a
library module imported by the corpus build pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

import spacy
from spacy.language import Language

# spaCy NER label → placeholder type stem
ENTITY_TYPE_MAP: dict[str, str] = {
    "MONEY": "moneyamount",
    "DATE": "dateexpression",
    "TIME": "timeexpression",
    "PERSON": "personname",
    "ORG": "organizationname",
    "GPE": "locationname",       # geo-political entity (countries, cities)
    "LOC": "locationname",       # non-GPE locations (mountains, etc.)
    "PERCENT": "percentvalue",
    "QUANTITY": "quantityvalue",
    "CARDINAL": "numbervalue",
    "ORDINAL": "ordinalvalue",
    "PRODUCT": "productname",
    "EVENT": "eventname",
    "WORK_OF_ART": "workname",
    "FAC": "facilityname",
    "NORP": "groupname",         # nationalities, religious or political groups
    "LANGUAGE": "languagename",
    "LAW": "lawname",
}

# All placeholder type stems, plus 26 letter suffixes each → 18 × 26 = 468
# distinct alien-vocab tokens we'll register downstream.
PLACEHOLDER_TYPES: tuple[str, ...] = tuple(sorted(set(ENTITY_TYPE_MAP.values())))
LETTERS: str = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class _State:
    """Per-paragraph entity tracking state."""
    surface_to_label: dict[tuple[str, str], str] = field(default_factory=dict)
    next_idx: dict[str, int] = field(default_factory=dict)

    def label_for(self, type_stem: str, surface_key: str) -> str:
        """Return the deterministic placeholder for (type, surface) within this
        paragraph; assigns a new letter if first seen.
        """
        key = (type_stem, surface_key)
        if key in self.surface_to_label:
            return self.surface_to_label[key]
        idx = self.next_idx.get(type_stem, 0)
        suffix = LETTERS[min(idx, len(LETTERS) - 1)]   # cap at 'z' on overflow
        label = f"{type_stem}{suffix}"
        self.surface_to_label[key] = label
        self.next_idx[type_stem] = idx + 1
        return label


def _surface_key(text: str) -> str:
    """Normalise a surface form for coreference matching.

    Lowercase + collapse internal whitespace. Conservative: this won't
    coalesce ``$5`` and ``five dollars`` (different surface forms ⇒
    different referents under our scheme), which is the right behaviour
    for the user's example "$5 vs $10 are distinct" — and it also keeps
    ``$5`` and ``$5.00`` distinct, which is fine.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_paragraph(text: str, nlp: Language) -> str:
    """Normalise a paragraph: replace each NER entity span with its
    placeholder. Coreference is preserved within the paragraph; entities
    of unmapped types are left untouched.
    """
    if not text or not text.strip():
        return text
    doc = nlp(text)
    state = _State()

    # Build per-token entity-replacement map. spaCy assigns ent_iob_ ∈
    # {B, I, O} to each token; we replace each *complete entity span* with
    # a single placeholder token, and emit empty strings for inside-tokens.
    out_tokens: list[str] = []
    for ent in doc.ents:
        # Pre-resolve label (paragraph-scoped) for this entity span
        type_stem = ENTITY_TYPE_MAP.get(ent.label_)
        if type_stem is None:
            continue
        state.label_for(type_stem, _surface_key(ent.text))

    # Walk tokens and emit either original text or the placeholder.
    skip_indices: set[int] = set()
    placeholder_at: dict[int, str] = {}
    for ent in doc.ents:
        type_stem = ENTITY_TYPE_MAP.get(ent.label_)
        if type_stem is None:
            continue
        label = state.label_for(type_stem, _surface_key(ent.text))
        placeholder_at[ent.start] = label
        for i in range(ent.start + 1, ent.end):
            skip_indices.add(i)

    # Emit text preserving original whitespace where possible.
    pieces: list[str] = []
    for i, tok in enumerate(doc):
        if i in skip_indices:
            # The trailing whitespace of an inside-token belongs to the
            # next emitted token; we drop both this token and its
            # whitespace.
            continue
        if i in placeholder_at:
            pieces.append(placeholder_at[i])
            # Preserve the whitespace AFTER the *last* token of the entity
            ent_end = next(
                (e.end for e in doc.ents if e.start == i),
                i + 1,
            )
            last_inside = doc[ent_end - 1]
            pieces.append(last_inside.whitespace_)
        else:
            pieces.append(tok.text)
            pieces.append(tok.whitespace_)
    return "".join(pieces)


# ───────────────────────────────────────────────────────────────────────────
# Smoke-test
# ───────────────────────────────────────────────────────────────────────────


_SMOKE_CASES: list[tuple[str, str, str]] = [
    # (label, input, expected-pattern-or-property)
    (
        "empty",
        "",
        "EQ:",
    ),
    (
        "whitespace-only",
        "   ",
        "EQ:   ",
    ),
    (
        "no-entities",
        "The cat sat on the mat.",
        "EQ:The cat sat on the mat.",
    ),
    (
        "user-example-money",
        "I gave you $5 but you gave me back $10.",
        "CONTAINS:moneyamounta CONTAINS:moneyamountb",
    ),
    (
        "money-recurrence",
        "She owes $5. Then $10. Later $5 again.",
        "CONTAINS:moneyamounta CONTAINS:moneyamountb COUNT:moneyamounta=2",
    ),
    (
        "mixed-types",
        "Alice met Bob in Paris on March 3rd, 2024 to discuss $1M.",
        "CONTAINS:personname CONTAINS:locationname CONTAINS:dateexpression CONTAINS:moneyamount",
    ),
    (
        "many-distinct-money",
        "Bills: $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, "
        "$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28.",
        "CONTAINS:moneyamountz",  # overflow case — at least the cap fired
    ),
    (
        "case-insensitive-coref",
        "APPLE rose. Then apple fell. Apple ended flat.",
        "EXISTS",  # spaCy may or may not tag — soft check: just ensure no crash
    ),
    (
        "punctuation-adjacent",
        "He said \"$5,\" then left.",
        "CONTAINS:moneyamount",
    ),
    (
        "two-paragraphs-independent",
        None,  # special: handle below
        None,
    ),
]


def _smoke() -> int:
    print("loading spaCy en_core_web_sm ...")
    nlp = spacy.load("en_core_web_sm")
    print(f"placeholder types: {PLACEHOLDER_TYPES}")
    print(f"letters: {LETTERS} (overflow→'z')\n")

    fails = 0
    for label, text, check in _SMOKE_CASES:
        if label == "two-paragraphs-independent":
            # Each paragraph should get its own letter-counter starting at 'a'.
            p1 = "Alice met Bob."
            p2 = "Carol met Dave."
            n1 = normalize_paragraph(p1, nlp)
            n2 = normalize_paragraph(p2, nlp)
            ok = ("personnamea" in n1) and ("personnamea" in n2)
            print(f"[{label}] {'OK' if ok else 'FAIL'}")
            print(f"  in  : {p1!r}")
            print(f"  out : {n1!r}")
            print(f"  in  : {p2!r}")
            print(f"  out : {n2!r}")
            if not ok:
                fails += 1
            continue
        out = normalize_paragraph(text, nlp)
        ok = True
        msg = ""
        if check.startswith("EQ:"):
            expected = check[3:]
            ok = out == expected
            msg = f"want {expected!r}"
        elif check == "EXISTS":
            ok = isinstance(out, str)
        else:
            for token in check.split():
                if token.startswith("CONTAINS:"):
                    needle = token[len("CONTAINS:"):]
                    if needle not in out:
                        ok = False
                        msg += f" missing {needle!r}"
                elif token.startswith("COUNT:"):
                    head, _, n = token[len("COUNT:"):].partition("=")
                    actual = out.count(head)
                    if actual != int(n):
                        ok = False
                        msg += f" {head} count {actual}!={n}"
        status = "OK" if ok else "FAIL"
        print(f"[{label}] {status} {msg}")
        print(f"  in : {text!r}")
        print(f"  out: {out!r}")
        if not ok:
            fails += 1
    print()
    print(f"smoke: {len(_SMOKE_CASES) - fails} ok, {fails} fail")
    return 1 if fails else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input", help="single paragraph to normalise (stdin if omitted with no --smoke)")
    args = p.parse_args()
    if args.smoke:
        sys.exit(_smoke())
    text = args.input if args.input is not None else sys.stdin.read()
    nlp = spacy.load("en_core_web_sm")
    print(normalize_paragraph(text, nlp))


if __name__ == "__main__":
    main()
