#!/usr/bin/env python3
"""Index linter for SCS LaTeX: fails the build when a per-trial SOLUTION symbol
drops its trial index. This is the self-running gate — it does not depend on a human
remembering to check. Wire it before pdflatex (see Makefile).

A per-trial solution (perturbation, coefficient, per-channel waveform, first-success
count, attacked trial) MUST carry the trial index i. The OPERATOR forms that legitimately
have no i (the waveform map E_{S,A}, the generic coefficient a_{c,m} inside that map) are
allowed only on a line explicitly tagged `% noqa: idx` — which forces an explicit
"yes, this is the operator" decision at the site, instead of a silent ambiguity.

Usage:  python3 check_indices.py FILE.tex [FILE2.tex ...]
Exit:   0 = clean, 1 = violations found (prints file:line: pattern).
"""
import re
import sys

# (name, regex) — each matches a per-trial solution written WITHOUT a leading trial index i.
# Good forms (E_{i,c}, a_{i,c,m}, k_i^\star, \mathbf{A}_i^{(k)}) deliberately do NOT match.
FORBIDDEN = [
    ("perturbation/waveform missing i", re.compile(r"E_c\b|E_c\(|E_\{c[,}]")),
    ("bold perturbation missing i",     re.compile(r"\\mathbf\{E\}_c\b|\\mathbf\{E\}_\{c[,}]")),
    ("coefficient missing i",           re.compile(r"a_\{c[,}]")),
    ("coeff matrix missing i",          re.compile(r"\\mathbf\{A\}\^\{\(")),       # A^{(k)} with no _i
    ("first-success count missing i",   re.compile(r"(?<![A-Za-z_^{])k\^\\?\{?\\star|k\^\*")),
]

NOQA = re.compile(r"%\s*noqa:\s*idx")


def strip_comment(line: str) -> str:
    """Drop the LaTeX comment tail (unescaped %), so commented-out math isn't linted."""
    out, esc = [], False
    for ch in line:
        if ch == "%" and not esc:
            break
        out.append(ch)
        esc = (ch == "\\") and not esc
    return "".join(out)


def lint(path: str) -> int:
    violations = 0
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            if NOQA.search(raw):
                continue
            code = strip_comment(raw)
            for name, rx in FORBIDDEN:
                if rx.search(code):
                    print(f"{path}:{n}: {name}  ->  {code.strip()[:90]}")
                    violations += 1
    return violations


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    total = sum(lint(p) for p in argv[1:])
    if total:
        print(f"\nINDEX LINT FAILED: {total} violation(s). "
              f"A per-trial solution dropped its index, or tag the operator line `% noqa: idx`.")
        return 1
    print("index lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
