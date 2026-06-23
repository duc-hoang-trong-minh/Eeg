from __future__ import annotations

from .greedy_attack import ChannelFirstScoreAttack


class QeldbaScoreAttack(ChannelFirstScoreAttack):
    """Paper-guided reconstruction of QELDBA (Qian et al., 2025) for the CCA harness.

    QELDBA is a query-efficient, likelihood-free black-box attack on EEG brainprint
    recognition that injects high-frequency perturbations. This is not an official
    upstream implementation; it is a faithful reconstruction of the method's
    distinctive ingredients so it can be compared against SCHS on the same score-only
    channel-control audit:

    * score-only (black-box) access, with no model gradients -- unlike the SAGA-style
      PGD reconstruction, which uses white-box input gradients;
    * a high-frequency perturbation unit: the per-channel waveform is parameterized on
      a sparse bank of high-frequency oscillatory atoms (configured via ``basis_mode``,
      ``basis_min_hz``/``basis_max_hz``), rather than the broadband 2--30 Hz basis SCHS
      uses;
    * a fixed query budget, matching the query-efficiency emphasis of the method.

    To avoid conflating QELDBA's high-frequency perturbation unit with optimizer
    strength, the search procedure (greedy channel selection + SPSA waveform
    refinement under the same peak-ratio cap and query budget) is held identical to
    SCHS. The only deliberate difference is the high-frequency perturbation basis, so
    the reported ASR / first-success channel budget gap is attributable to the
    perturbation unit and not to a weaker optimizer.
    """
