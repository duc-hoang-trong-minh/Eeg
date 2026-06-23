# Research gate checklist

ONE RULE makes this work: each gate is a HARD STOP. You may not write anything
belonging to a later gate until every box in the current gate is checked AND written
down. The ordering kills the pipeline error (answer-before-framing); the items kill
the rest. No checklist is literally fool-proof — the discipline is stopping at each
gate even when you already think you know the answer.

See ~/.claude/CLAUDE.md for the reasoning behind each gate.

## GATE 0 — Question   (kills: solution-first, answer-before-framing)
- [ ] Question written as an OPEN question, no answer baked in. EEG could be use as an identifier, what is the most realistic way to attack it?
- [ ] "Why is it open?" written — the naive answer and why it fails. (blank => trivial) To attack realisticly we do blackbox search which is very very hard be cause of the channelxtime search space is huge. Qurey it would takes times.
- [ ] Method/answer NOT written anywhere yet.

## GATE 1 — Assumptions   (kills: unclear / convenient assumptions)
- [ ] Each assumption written falsifiably: "we assume X." We assume blackbox adversarial attack, we dont know the training dataset, architecture or weight of the model, still have to attack. We only have one dataset D to query on the attack.
- [ ] Convenience test per assumption: "would I have written this if I did NOT already  know my method?" If no -> rewrite or drop. Yes, check the adversarial attack blackbox setting.
- [ ] Each checked against the real setting: true of EEG/the model, or just for me? In reality, if you tamper with a eeg, you kinda dont know the model. So blackobx is safe here. Dont know if i am missing anythings?

## GATE 2 — System model / notation   (kills: loose notation, cascading missing detail) Shouldnt you do this and me check?
- [ ] Every object declared with (set, index, dimension) before any equation.

- [ ] Every data sample carries its index, unless I EXPLICITLY wrote "fix one point, drop the index."
- [ ] DEPENDENCY-LEDGER test: every symbol has a `depends on` entry before use.
      For each equation, compare dependencies on both sides. If the RHS depends on trial
      `i`, channel `c`, atom `m`, support size `k`, probe sample count `n`, probe draw
      `j`, or optimization step `ell`, the LHS must show that dependence or the equation
      is rejected. Do not rely on prose to carry hidden dependencies.
- [ ] AGGREGATION-INDEX test: for every de-indexed symbol, is it ever summed/averaged/
      max'd/ranked/compared across the data later (a metric, expectation, ranking)? If yes,
      the index STAYS. Hatch only for operators (loss, parameterization, update), never for
      the per-point solution they produce (chosen perturbation, its support, per-point outcome).
- [ ] DYNAMIC-OBJECT test: a set/sequence built incrementally is written S^{(1)} ⊂ S^{(2)} ⊂ ...
      with its growth rule, NOT a static S. An object chosen FOR a point is a function of it (E_i).
- [ ] BLAST-RADIUS test: after indexing the dataset, did the index propagate to EVERY object
      derived from a data point? Stopping at the declaration = a patch, not a refactor.
- [ ] Notation ledger started (symbol -> type, index range, dim, meaning).
- [ ] System model written independent of the solution.

## GATE 3 — Claim + proof   (kills: math full of holes, doing too much at once)
- [ ] ONE claim only. Statement written with quantifiers.
- [ ] Every proof step labeled proven / standard(+ref) / assumption / gap.
- [ ] Every equation type-checked: membership, indices bound, dimensions conform.
- [ ] No gap silently called proven. Lowest lemma closed first.
- [ ] Did not start a second claim before this one is airtight.

## GATE 3b — Proof audit: logic gaps & undefined stuff
Run after the claim is drafted, before it counts as done. Build an ENTAILMENT LEDGER:
one row per proof line -> {prior lines it uses} + {named rule/theorem that licenses it}.

A. Definition closure (undefined stuff)
- [ ] Every symbol in every equation traces to a line in the notation ledger. Symbol
      not in ledger = undefined, stop.
- [ ] No symbol used before its declaration line.
- [ ] Every argmax/argmin/sup/inf/inverse/conditional-expectation/density is
      WELL-DEFINED (exists, unique or tie-break stated, conditioning event non-null).
- [ ] Every denominator != 0; every log/sqrt domain valid; every set picked from non-empty.

B. Entailment closure (logic gaps)
- [ ] Each step cites its premises: which prior lines + which rule licenses the jump.
      Uncited "therefore" = gap.
- [ ] No circularity: no step uses the claim or a downstream consequence.
- [ ] Quantifier scope: each variable bound (forall/exists stated) or explicitly fixed;
      no universally-quantified var used as one specific value.
- [ ] Inequality direction preserved through every operation.
- [ ] Every limit/sum/expectation/integral interchange is licensed (DCT/Fubini/
      uniform convergence), not silently swapped.

C. Assumption discharge (hidden assumptions)
- [ ] Every assumption USED in the proof appears in the claim's assumption list.
- [ ] Every assumption LISTED is actually used somewhere.

## GATE 4 — Important parameters   (kills: "we tested, got a number, we're happy")
- [ ] For each parameter the result is sensitive to: governing quantity named.
- [ ] Scaling law / threshold DERIVED before the sweep.
- [ ] Sweep written to CONFIRM that law, not to find a knee.

## GATE 5 — Experiment -> insight   (kills: number without insight)
- [ ] Every number answered with a MECHANISM, not "it works."
- [ ] Experiment came AFTER the theory (check git order).

## GATE 6 — Motivation + writing   (kills: unclear motivation, weak writing)
- [ ] Motivation written as a gap: "work assumes A; in our setting A fails because B; therefore C." We found a way to efficiently probe the 
- [ ] Section order: question -> assumptions -> claim -> proof -> experiment -> insight.
