# Build with the index gate in front of LaTeX.
# `make tech` / `make paper` run the linter FIRST; pdflatex never runs on a violation.
# This is the enforcement: the check runs itself, not when someone remembers to.

LINT = python3 check_indices.py

.PHONY: tech paper lint clean

tech: lint-tech
	pdflatex -interaction=nonstopmode -halt-on-error TECHNICAL.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error TECHNICAL.tex >/dev/null

paper: lint-paper
	pdflatex -interaction=nonstopmode Paper_draft.tex >/dev/null
	pdflatex -interaction=nonstopmode Paper_draft.tex >/dev/null

lint-tech:
	$(LINT) TECHNICAL.tex

lint-paper:
	$(LINT) Paper_draft.tex

lint:
	$(LINT) TECHNICAL.tex Paper_draft.tex

clean:
	rm -f *.aux *.log *.out
