from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .basis import build_basis_matrix, synthesize_window_perturbation
from .losses import band_energy_penalty, tv_regularizer, untargeted_margin
from .spsa import spsa_minimize
from .support import WindowPartition, all_atoms, make_window_partition


ScoreFn = Callable[[np.ndarray], np.ndarray]


class QueryBudgetExhausted(RuntimeError):
    pass


@dataclass
class AttackResult:
    x_adv: np.ndarray
    delta: np.ndarray
    support: list[tuple[int, int] | int]
    coeffs: np.ndarray
    margin: float
    success: bool
    queries_used: int
    budget_exhausted: bool


def _apply_peak_ratio_constraint(
    x: np.ndarray,
    delta: np.ndarray,
    max_perturbation_peak_ratio: float | None,
) -> np.ndarray:
    if max_perturbation_peak_ratio is None:
        return delta
    signal_peak = float(np.max(np.abs(x)))
    if signal_peak <= 0.0:
        return delta
    allowed_peak = float(max_perturbation_peak_ratio) * signal_peak
    delta_peak = float(np.max(np.abs(delta)))
    if delta_peak <= allowed_peak or delta_peak <= 0.0:
        return delta
    scale = allowed_peak / delta_peak
    return (delta * scale).astype(np.float32, copy=False)


def _resolve_channel_coeff_rank(
    basis_rank_r: int,
    n_windows: int,
    channel_waveform_rank: int | None,
) -> int:
    if channel_waveform_rank is not None:
        return int(channel_waveform_rank)
    return max(int(basis_rank_r * n_windows), int(basis_rank_r))


def _build_full_trial_basis_matrix(
    basis_mode: str,
    n_samples: int,
    coeff_rank: int,
    basis_min_hz: float,
    basis_max_hz: float,
    sfreq: float,
    basis_phase_count: int,
) -> np.ndarray:
    full_basis_mode = basis_mode if basis_mode in {"hybrid", "freq_atom_bank", "raised_cosine"} else "hybrid"
    return build_basis_matrix(
        basis_mode=full_basis_mode,
        window_length=n_samples,
        rank=coeff_rank,
        f_min_hz=basis_min_hz,
        f_max_hz=basis_max_hz,
        sfreq=sfreq,
        phase_count=basis_phase_count,
    )


class GreedySparseScoreAttack:
    def __init__(
        self,
        score_fn: ScoreFn,
        sfreq: float,
        n_windows: int,
        support_budget_k: int,
        basis_rank_r: int,
        basis_min_hz: float,
        basis_max_hz: float,
        basis_mode: str,
        basis_phase_count: int,
        candidate_probe_restarts: int,
        candidate_probe_scale: float,
        max_outer_iters: int,
        max_query_budget: int | None,
        spsa_steps: int,
        spsa_step_size: float,
        spsa_perturb_scale: float,
        spsa_restarts: int,
        spsa_init_scale: float,
        l2_weight: float,
        tv_weight: float,
        band_weight: float,
        max_coeff_abs: float,
        max_perturbation_peak_ratio: float | None,
        enforce_unique_channels: bool = False,
        stop_on_success: bool = True,
        seed: int = 0,
    ):
        self.score_fn = score_fn
        self.sfreq = sfreq
        self.n_windows = n_windows
        self.support_budget_k = support_budget_k
        self.basis_rank_r = basis_rank_r
        self.basis_min_hz = basis_min_hz
        self.basis_max_hz = basis_max_hz
        self.basis_mode = basis_mode
        self.basis_phase_count = basis_phase_count
        self.candidate_probe_restarts = candidate_probe_restarts
        self.candidate_probe_scale = candidate_probe_scale
        self.max_outer_iters = max_outer_iters
        self.max_query_budget = max_query_budget
        self.spsa_steps = spsa_steps
        self.spsa_step_size = spsa_step_size
        self.spsa_perturb_scale = spsa_perturb_scale
        self.spsa_restarts = spsa_restarts
        self.spsa_init_scale = spsa_init_scale
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
        self.band_weight = band_weight
        self.max_coeff_abs = max_coeff_abs
        self.max_perturbation_peak_ratio = max_perturbation_peak_ratio
        self.enforce_unique_channels = enforce_unique_channels
        self.stop_on_success = stop_on_success
        self.seed = seed
        self.queries_used = 0

    def _query_scores(self, x: np.ndarray) -> np.ndarray:
        if self.max_query_budget is not None and self.queries_used >= self.max_query_budget:
            raise QueryBudgetExhausted(
                f"Query budget exhausted at {self.queries_used} / {self.max_query_budget} queries."
            )
        self.queries_used += 1
        return self.score_fn(x)

    def _assemble_delta(
        self,
        n_channels: int,
        n_samples: int,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
    ) -> np.ndarray:
        delta = np.zeros((n_channels, n_samples), dtype=np.float32)
        for (c, w), atom_coeffs in zip(support, coeffs):
            s, e = partition.boundaries[w]
            local = synthesize_window_perturbation(atom_coeffs, basis_by_window[w])
            delta[c, s:e] += local.astype(np.float32)
        return delta

    def _objective(
        self,
        x: np.ndarray,
        y: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
    ) -> float:
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
        )
        delta = _apply_peak_ratio_constraint(x=x, delta=delta, max_perturbation_peak_ratio=self.max_perturbation_peak_ratio)
        x_adv = x + delta
        scores = self._query_scores(x_adv)
        margin = untargeted_margin(scores, y)
        l2 = float(np.mean(coeffs**2))
        tv = tv_regularizer(delta)
        band = band_energy_penalty(delta, sfreq=self.sfreq)
        return margin + self.l2_weight * l2 + self.tv_weight * tv + self.band_weight * band

    def _build_result(
        self,
        x: np.ndarray,
        y: int,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        margin: float,
        budget_exhausted: bool,
    ) -> AttackResult:
        aligned_support = support[: len(coeffs)]
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            partition=partition,
            basis_by_window=basis_by_window,
            support=aligned_support,
            coeffs=coeffs,
        )
        delta = _apply_peak_ratio_constraint(x=x, delta=delta, max_perturbation_peak_ratio=self.max_perturbation_peak_ratio)
        return AttackResult(
            x_adv=x + delta,
            delta=delta,
            support=aligned_support,
            coeffs=coeffs.copy(),
            margin=margin,
            success=margin < 0.0,
            queries_used=self.queries_used,
            budget_exhausted=budget_exhausted,
        )

    def _refine_coeffs(
        self,
        x: np.ndarray,
        y: int,
        support: list[tuple[int, int]],
        init_coeffs: np.ndarray,
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        if len(support) == 0:
            scores = self._query_scores(x)
            return init_coeffs, untargeted_margin(scores, y)

        flat0 = init_coeffs.reshape(-1)

        def f(flat: np.ndarray) -> float:
            coeffs = flat.reshape(len(support), self.basis_rank_r)
            return self._objective(x, y, support, coeffs, partition, basis_by_window)

        flat_best, value = spsa_minimize(
            objective=f,
            x0=flat0,
            steps=self.spsa_steps,
            step_size=self.spsa_step_size,
            perturb_scale=self.spsa_perturb_scale,
            clip_abs=self.max_coeff_abs,
            restarts=self.spsa_restarts,
            init_scale=self.spsa_init_scale,
            seed=self.seed,
        )
        return flat_best.reshape(len(support), self.basis_rank_r), value

    def _estimate_candidate(
        self,
        x: np.ndarray,
        y: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        atom: tuple[int, int],
        partition: WindowPartition,
        basis_by_window: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        candidate_support = support + [atom]
        base_coeffs = np.vstack([coeffs, np.zeros((1, self.basis_rank_r), dtype=np.float32)])
        best_coeffs = base_coeffs
        best_value = float(self._objective(x, y, candidate_support, base_coeffs, partition, basis_by_window))

        probe_abs = float(self.candidate_probe_scale * self.max_coeff_abs)
        if probe_abs <= 0.0 or self.candidate_probe_restarts <= 0:
            return best_coeffs, best_value

        for _ in range(self.candidate_probe_restarts):
            direction = rng.choice([-1.0, 1.0], size=(self.basis_rank_r,)).astype(np.float32)
            for sign in (-1.0, 1.0):
                proposal = base_coeffs.copy()
                proposal[-1] = sign * probe_abs * direction
                value = float(
                    self._objective(
                        x,
                        y,
                        candidate_support,
                        proposal,
                        partition,
                        basis_by_window,
                    )
                )
                if value < best_value:
                    best_value = value
                    best_coeffs = proposal

        return best_coeffs, best_value

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        self.queries_used = 0
        n_channels, n_samples = x.shape
        partition = make_window_partition(n_samples, self.n_windows)
        rng = np.random.default_rng(self.seed)

        basis_by_window = {}
        for w, (s, e) in enumerate(partition.boundaries):
            basis_by_window[w] = build_basis_matrix(
                basis_mode=self.basis_mode,
                window_length=e - s,
                rank=self.basis_rank_r,
                f_min_hz=self.basis_min_hz,
                f_max_hz=self.basis_max_hz,
                sfreq=self.sfreq,
                phase_count=self.basis_phase_count,
            )

        support: list[tuple[int, int]] = []
        coeffs = np.zeros((0, self.basis_rank_r), dtype=np.float32)

        initial_scores = self._query_scores(x)
        current_margin = untargeted_margin(initial_scores, y)
        if current_margin < 0.0:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=False,
            )

        universe = all_atoms(n_channels, partition)
        selected = set()
        n_outer_iters = min(self.support_budget_k, self.max_outer_iters)

        try:
            for _ in range(n_outer_iters):
                best_candidate = None
                best_value = float("inf")
                best_candidate_init = None

                for atom in universe:
                    if atom in selected:
                        continue
                    if self.enforce_unique_channels and any(int(existing_c) == int(atom[0]) for existing_c, _ in support):
                        continue
                    candidate_coeffs, value = self._estimate_candidate(
                        x=x,
                        y=y,
                        support=support,
                        coeffs=coeffs,
                        atom=atom,
                        partition=partition,
                        basis_by_window=basis_by_window,
                        rng=rng,
                    )
                    if value < best_value:
                        best_value = value
                        best_candidate = atom
                        best_candidate_init = candidate_coeffs

                if best_candidate is None or best_candidate_init is None:
                    break

                support.append(best_candidate)
                selected.add(best_candidate)
                coeffs, _ = self._refine_coeffs(
                    x=x,
                    y=y,
                    support=support,
                    init_coeffs=best_candidate_init,
                    partition=partition,
                    basis_by_window=basis_by_window,
                )

                delta = self._assemble_delta(
                    n_channels=n_channels,
                    n_samples=n_samples,
                    partition=partition,
                    basis_by_window=basis_by_window,
                    support=support,
                    coeffs=coeffs,
                )
                delta = _apply_peak_ratio_constraint(
                    x=x,
                    delta=delta,
                    max_perturbation_peak_ratio=self.max_perturbation_peak_ratio,
                )
                current_margin = untargeted_margin(self._query_scores(x + delta), y)
                if current_margin < 0.0 and self.stop_on_success:
                    return self._build_result(
                        x=x,
                        y=y,
                        partition=partition,
                        basis_by_window=basis_by_window,
                        support=support,
                        coeffs=coeffs,
                        margin=current_margin,
                        budget_exhausted=False,
                    )
        except QueryBudgetExhausted:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=True,
            )

        return self._build_result(
            x=x,
            y=y,
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
            margin=current_margin,
            budget_exhausted=False,
        )


class ChannelFirstScoreAttack:
    def __init__(
        self,
        score_fn: ScoreFn,
        sfreq: float,
        n_windows: int,
        support_budget_k: int,
        basis_rank_r: int,
        basis_min_hz: float,
        basis_max_hz: float,
        basis_mode: str,
        basis_phase_count: int,
        candidate_probe_restarts: int,
        candidate_probe_scale: float,
        max_outer_iters: int,
        max_query_budget: int | None,
        spsa_steps: int,
        spsa_step_size: float,
        spsa_perturb_scale: float,
        spsa_restarts: int,
        spsa_init_scale: float,
        l2_weight: float,
        tv_weight: float,
        band_weight: float,
        max_coeff_abs: float,
        max_perturbation_peak_ratio: float | None,
        channel_waveform_rank: int | None = None,
        enforce_unique_channels: bool = False,
        stop_on_success: bool = True,
        seed: int = 0,
    ):
        self.score_fn = score_fn
        self.sfreq = sfreq
        self.n_windows = n_windows
        self.support_budget_k = support_budget_k
        self.basis_rank_r = basis_rank_r
        self.basis_min_hz = basis_min_hz
        self.basis_max_hz = basis_max_hz
        self.basis_mode = basis_mode
        self.basis_phase_count = basis_phase_count
        self.candidate_probe_restarts = candidate_probe_restarts
        self.candidate_probe_scale = candidate_probe_scale
        self.max_outer_iters = max_outer_iters
        self.max_query_budget = max_query_budget
        self.spsa_steps = spsa_steps
        self.spsa_step_size = spsa_step_size
        self.spsa_perturb_scale = spsa_perturb_scale
        self.spsa_restarts = spsa_restarts
        self.spsa_init_scale = spsa_init_scale
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
        self.band_weight = band_weight
        self.max_coeff_abs = max_coeff_abs
        self.max_perturbation_peak_ratio = max_perturbation_peak_ratio
        self.channel_waveform_rank = channel_waveform_rank
        self.enforce_unique_channels = enforce_unique_channels
        self.stop_on_success = stop_on_success
        self.seed = seed
        self.queries_used = 0

    def _query_scores(self, x: np.ndarray) -> np.ndarray:
        if self.max_query_budget is not None and self.queries_used >= self.max_query_budget:
            raise QueryBudgetExhausted(
                f"Query budget exhausted at {self.queries_used} / {self.max_query_budget} queries."
            )
        self.queries_used += 1
        return self.score_fn(x)

    def _coeff_rank(self) -> int:
        return _resolve_channel_coeff_rank(
            basis_rank_r=self.basis_rank_r,
            n_windows=self.n_windows,
            channel_waveform_rank=self.channel_waveform_rank,
        )

    def _assemble_delta(
        self,
        n_channels: int,
        n_samples: int,
        basis_matrix: np.ndarray,
        support: list[int],
        coeffs: np.ndarray,
    ) -> np.ndarray:
        delta = np.zeros((n_channels, n_samples), dtype=np.float32)
        for channel, atom_coeffs in zip(support, coeffs):
            waveform = synthesize_window_perturbation(atom_coeffs, basis_matrix)
            delta[channel, :] += waveform.astype(np.float32)
        return delta

    def _objective(
        self,
        x: np.ndarray,
        y: int,
        support: list[int],
        coeffs: np.ndarray,
        basis_matrix: np.ndarray,
    ) -> float:
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            basis_matrix=basis_matrix,
            support=support,
            coeffs=coeffs,
        )
        delta = _apply_peak_ratio_constraint(x=x, delta=delta, max_perturbation_peak_ratio=self.max_perturbation_peak_ratio)
        scores = self._query_scores(x + delta)
        margin = untargeted_margin(scores, y)
        l2 = float(np.mean(coeffs**2))
        tv = tv_regularizer(delta)
        band = band_energy_penalty(delta, sfreq=self.sfreq)
        return margin + self.l2_weight * l2 + self.tv_weight * tv + self.band_weight * band

    def _build_result(
        self,
        x: np.ndarray,
        y: int,
        basis_matrix: np.ndarray,
        support: list[int],
        coeffs: np.ndarray,
        margin: float,
        budget_exhausted: bool,
    ) -> AttackResult:
        aligned_support = support[: len(coeffs)]
        delta = self._assemble_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            basis_matrix=basis_matrix,
            support=aligned_support,
            coeffs=coeffs,
        )
        delta = _apply_peak_ratio_constraint(x=x, delta=delta, max_perturbation_peak_ratio=self.max_perturbation_peak_ratio)
        return AttackResult(
            x_adv=x + delta,
            delta=delta,
            support=aligned_support,
            coeffs=coeffs.copy(),
            margin=margin,
            success=margin < 0.0,
            queries_used=self.queries_used,
            budget_exhausted=budget_exhausted,
        )

    def _refine_coeffs(
        self,
        x: np.ndarray,
        y: int,
        support: list[int],
        init_coeffs: np.ndarray,
        basis_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        coeff_rank = self._coeff_rank()
        if len(support) == 0:
            scores = self._query_scores(x)
            return init_coeffs, untargeted_margin(scores, y)

        flat0 = init_coeffs.reshape(-1)

        def f(flat: np.ndarray) -> float:
            coeffs = flat.reshape(len(support), coeff_rank)
            return self._objective(x, y, support, coeffs, basis_matrix)

        flat_best, value = spsa_minimize(
            objective=f,
            x0=flat0,
            steps=self.spsa_steps,
            step_size=self.spsa_step_size,
            perturb_scale=self.spsa_perturb_scale,
            clip_abs=self.max_coeff_abs,
            restarts=self.spsa_restarts,
            init_scale=self.spsa_init_scale,
            seed=self.seed,
        )
        return flat_best.reshape(len(support), coeff_rank), value

    def _estimate_candidate(
        self,
        x: np.ndarray,
        y: int,
        support: list[int],
        coeffs: np.ndarray,
        channel: int,
        basis_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        coeff_rank = self._coeff_rank()
        candidate_support = support + [channel]
        base_coeffs = np.vstack([coeffs, np.zeros((1, coeff_rank), dtype=np.float32)])
        best_coeffs = base_coeffs
        best_value = float(self._objective(x, y, candidate_support, base_coeffs, basis_matrix))

        probe_abs = float(self.candidate_probe_scale * self.max_coeff_abs)
        if probe_abs <= 0.0 or self.candidate_probe_restarts <= 0:
            return best_coeffs, best_value

        for _ in range(self.candidate_probe_restarts):
            direction = rng.choice([-1.0, 1.0], size=(coeff_rank,)).astype(np.float32)
            for sign in (-1.0, 1.0):
                proposal = base_coeffs.copy()
                proposal[-1] = sign * probe_abs * direction
                value = float(self._objective(x, y, candidate_support, proposal, basis_matrix))
                if value < best_value:
                    best_value = value
                    best_coeffs = proposal

        return best_coeffs, best_value

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        self.queries_used = 0
        n_channels, n_samples = x.shape
        rng = np.random.default_rng(self.seed)

        basis_matrix = _build_full_trial_basis_matrix(
            basis_mode=self.basis_mode,
            n_samples=n_samples,
            coeff_rank=self._coeff_rank(),
            basis_min_hz=self.basis_min_hz,
            basis_max_hz=self.basis_max_hz,
            sfreq=self.sfreq,
            basis_phase_count=self.basis_phase_count,
        )

        support: list[int] = []
        coeffs = np.zeros((0, self._coeff_rank()), dtype=np.float32)

        initial_scores = self._query_scores(x)
        current_margin = untargeted_margin(initial_scores, y)
        if current_margin < 0.0:
            return self._build_result(
                x=x,
                y=y,
                basis_matrix=basis_matrix,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=False,
            )

        universe = list(range(n_channels))
        selected: set[int] = set()
        n_outer_iters = min(self.support_budget_k, self.max_outer_iters)

        try:
            for _ in range(n_outer_iters):
                best_channel = None
                best_value = float("inf")
                best_candidate_init = None

                for channel in universe:
                    if channel in selected:
                        continue
                    candidate_coeffs, value = self._estimate_candidate(
                        x=x,
                        y=y,
                        support=support,
                        coeffs=coeffs,
                        channel=channel,
                        basis_matrix=basis_matrix,
                        rng=rng,
                    )
                    if value < best_value:
                        best_value = value
                        best_channel = channel
                        best_candidate_init = candidate_coeffs

                if best_channel is None or best_candidate_init is None:
                    break

                support.append(best_channel)
                selected.add(best_channel)
                coeffs, _ = self._refine_coeffs(
                    x=x,
                    y=y,
                    support=support,
                    init_coeffs=best_candidate_init,
                    basis_matrix=basis_matrix,
                )

                delta = self._assemble_delta(
                    n_channels=n_channels,
                    n_samples=n_samples,
                    basis_matrix=basis_matrix,
                    support=support,
                    coeffs=coeffs,
                )
                delta = _apply_peak_ratio_constraint(
                    x=x,
                    delta=delta,
                    max_perturbation_peak_ratio=self.max_perturbation_peak_ratio,
                )
                current_margin = untargeted_margin(self._query_scores(x + delta), y)
                if current_margin < 0.0 and self.stop_on_success:
                    return self._build_result(
                        x=x,
                        y=y,
                        basis_matrix=basis_matrix,
                        support=support,
                        coeffs=coeffs,
                        margin=current_margin,
                        budget_exhausted=False,
                    )
        except QueryBudgetExhausted:
            return self._build_result(
                x=x,
                y=y,
                basis_matrix=basis_matrix,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=True,
            )

        return self._build_result(
            x=x,
            y=y,
            basis_matrix=basis_matrix,
            support=support,
            coeffs=coeffs,
            margin=current_margin,
            budget_exhausted=False,
        )


class ChannelThenWindowScoreAttack(GreedySparseScoreAttack):
    def __init__(
        self,
        score_fn: ScoreFn,
        sfreq: float,
        n_windows: int,
        support_budget_k: int,
        basis_rank_r: int,
        basis_min_hz: float,
        basis_max_hz: float,
        basis_mode: str,
        basis_phase_count: int,
        candidate_probe_restarts: int,
        candidate_probe_scale: float,
        max_outer_iters: int,
        max_query_budget: int | None,
        spsa_steps: int,
        spsa_step_size: float,
        spsa_perturb_scale: float,
        spsa_restarts: int,
        spsa_init_scale: float,
        l2_weight: float,
        tv_weight: float,
        band_weight: float,
        max_coeff_abs: float,
        max_perturbation_peak_ratio: float | None,
        channel_waveform_rank: int | None = None,
        channel_shortlist_size: int | None = None,
        enforce_unique_channels: bool = False,
        stop_on_success: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            score_fn=score_fn,
            sfreq=sfreq,
            n_windows=n_windows,
            support_budget_k=support_budget_k,
            basis_rank_r=basis_rank_r,
            basis_min_hz=basis_min_hz,
            basis_max_hz=basis_max_hz,
            basis_mode=basis_mode,
            basis_phase_count=basis_phase_count,
            candidate_probe_restarts=candidate_probe_restarts,
            candidate_probe_scale=candidate_probe_scale,
            max_outer_iters=max_outer_iters,
            max_query_budget=max_query_budget,
            spsa_steps=spsa_steps,
            spsa_step_size=spsa_step_size,
            spsa_perturb_scale=spsa_perturb_scale,
            spsa_restarts=spsa_restarts,
            spsa_init_scale=spsa_init_scale,
            l2_weight=l2_weight,
            tv_weight=tv_weight,
            band_weight=band_weight,
            max_coeff_abs=max_coeff_abs,
            max_perturbation_peak_ratio=max_perturbation_peak_ratio,
            enforce_unique_channels=enforce_unique_channels,
            stop_on_success=stop_on_success,
            seed=seed,
        )
        self.channel_waveform_rank = channel_waveform_rank
        self.channel_shortlist_size = channel_shortlist_size

    def _channel_coeff_rank(self) -> int:
        return _resolve_channel_coeff_rank(
            basis_rank_r=self.basis_rank_r,
            n_windows=self.n_windows,
            channel_waveform_rank=self.channel_waveform_rank,
        )

    def _assemble_channel_delta(
        self,
        n_channels: int,
        n_samples: int,
        basis_matrix: np.ndarray,
        support: list[int],
        coeffs: np.ndarray,
    ) -> np.ndarray:
        delta = np.zeros((n_channels, n_samples), dtype=np.float32)
        for channel, atom_coeffs in zip(support, coeffs):
            waveform = synthesize_window_perturbation(atom_coeffs, basis_matrix)
            delta[channel, :] += waveform.astype(np.float32)
        return delta

    def _objective_channel(
        self,
        x: np.ndarray,
        y: int,
        support: list[int],
        coeffs: np.ndarray,
        basis_matrix: np.ndarray,
    ) -> float:
        delta = self._assemble_channel_delta(
            n_channels=x.shape[0],
            n_samples=x.shape[1],
            basis_matrix=basis_matrix,
            support=support,
            coeffs=coeffs,
        )
        delta = _apply_peak_ratio_constraint(
            x=x,
            delta=delta,
            max_perturbation_peak_ratio=self.max_perturbation_peak_ratio,
        )
        scores = self._query_scores(x + delta)
        margin = untargeted_margin(scores, y)
        l2 = float(np.mean(coeffs**2))
        tv = tv_regularizer(delta)
        band = band_energy_penalty(delta, sfreq=self.sfreq)
        return margin + self.l2_weight * l2 + self.tv_weight * tv + self.band_weight * band

    def _estimate_channel(
        self,
        x: np.ndarray,
        y: int,
        channel: int,
        basis_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        coeff_rank = self._channel_coeff_rank()
        support = [channel]
        base_coeffs = np.zeros((1, coeff_rank), dtype=np.float32)
        best_value = float(self._objective_channel(x, y, support, base_coeffs, basis_matrix))

        probe_abs = float(self.candidate_probe_scale * self.max_coeff_abs)
        if probe_abs <= 0.0 or self.candidate_probe_restarts <= 0:
            return best_value

        for _ in range(self.candidate_probe_restarts):
            direction = rng.choice([-1.0, 1.0], size=(coeff_rank,)).astype(np.float32)
            for sign in (-1.0, 1.0):
                proposal = base_coeffs.copy()
                proposal[-1] = sign * probe_abs * direction
                value = float(self._objective_channel(x, y, support, proposal, basis_matrix))
                if value < best_value:
                    best_value = value
        return best_value

    def _shortlist_channels(self, x: np.ndarray, y: int, rng: np.random.Generator) -> list[int]:
        n_channels, n_samples = x.shape
        shortlist_size = self.channel_shortlist_size
        if shortlist_size is None:
            shortlist_size = min(n_channels, max(self.support_budget_k * 2, self.support_budget_k))
        shortlist_size = max(1, min(int(shortlist_size), n_channels))
        if shortlist_size >= n_channels:
            return list(range(n_channels))

        basis_matrix = _build_full_trial_basis_matrix(
            basis_mode=self.basis_mode,
            n_samples=n_samples,
            coeff_rank=self._channel_coeff_rank(),
            basis_min_hz=self.basis_min_hz,
            basis_max_hz=self.basis_max_hz,
            sfreq=self.sfreq,
            basis_phase_count=self.basis_phase_count,
        )
        channel_scores = []
        for channel in range(n_channels):
            channel_value = self._estimate_channel(
                x=x,
                y=y,
                channel=channel,
                basis_matrix=basis_matrix,
                rng=rng,
            )
            channel_scores.append((channel_value, channel))
        channel_scores.sort(key=lambda row: row[0])
        return [channel for _, channel in channel_scores[:shortlist_size]]

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        self.queries_used = 0
        n_channels, n_samples = x.shape
        partition = make_window_partition(n_samples, self.n_windows)
        rng = np.random.default_rng(self.seed)

        basis_by_window = {}
        for w, (s, e) in enumerate(partition.boundaries):
            basis_by_window[w] = build_basis_matrix(
                basis_mode=self.basis_mode,
                window_length=e - s,
                rank=self.basis_rank_r,
                f_min_hz=self.basis_min_hz,
                f_max_hz=self.basis_max_hz,
                sfreq=self.sfreq,
                phase_count=self.basis_phase_count,
            )

        support: list[tuple[int, int]] = []
        coeffs = np.zeros((0, self.basis_rank_r), dtype=np.float32)

        initial_scores = self._query_scores(x)
        current_margin = untargeted_margin(initial_scores, y)
        if current_margin < 0.0:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=False,
            )

        shortlisted_channels = set(self._shortlist_channels(x=x, y=y, rng=rng))
        universe = [atom for atom in all_atoms(n_channels, partition) if atom[0] in shortlisted_channels]
        selected = set()
        n_outer_iters = min(self.support_budget_k, self.max_outer_iters)

        try:
            for _ in range(n_outer_iters):
                best_candidate = None
                best_value = float("inf")
                best_candidate_init = None

                for atom in universe:
                    if atom in selected:
                        continue
                    if self.enforce_unique_channels and any(int(existing_c) == int(atom[0]) for existing_c, _ in support):
                        continue
                    candidate_coeffs, value = self._estimate_candidate(
                        x=x,
                        y=y,
                        support=support,
                        coeffs=coeffs,
                        atom=atom,
                        partition=partition,
                        basis_by_window=basis_by_window,
                        rng=rng,
                    )
                    if value < best_value:
                        best_value = value
                        best_candidate = atom
                        best_candidate_init = candidate_coeffs

                if best_candidate is None or best_candidate_init is None:
                    break

                support.append(best_candidate)
                selected.add(best_candidate)
                coeffs, _ = self._refine_coeffs(
                    x=x,
                    y=y,
                    support=support,
                    init_coeffs=best_candidate_init,
                    partition=partition,
                    basis_by_window=basis_by_window,
                )

                delta = self._assemble_delta(
                    n_channels=n_channels,
                    n_samples=n_samples,
                    partition=partition,
                    basis_by_window=basis_by_window,
                    support=support,
                    coeffs=coeffs,
                )
                delta = _apply_peak_ratio_constraint(
                    x=x,
                    delta=delta,
                    max_perturbation_peak_ratio=self.max_perturbation_peak_ratio,
                )
                current_margin = untargeted_margin(self._query_scores(x + delta), y)
                if current_margin < 0.0:
                    return self._build_result(
                        x=x,
                        y=y,
                        partition=partition,
                        basis_by_window=basis_by_window,
                        support=support,
                        coeffs=coeffs,
                        margin=current_margin,
                        budget_exhausted=False,
                    )
        except QueryBudgetExhausted:
            return self._build_result(
                x=x,
                y=y,
                partition=partition,
                basis_by_window=basis_by_window,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=True,
            )

        return self._build_result(
            x=x,
            y=y,
            partition=partition,
            basis_by_window=basis_by_window,
            support=support,
            coeffs=coeffs,
            margin=current_margin,
            budget_exhausted=False,
        )


def build_score_attack(
    score_fn: ScoreFn,
    sfreq: float,
    n_windows: int,
    support_budget_k: int,
    basis_rank_r: int,
    basis_min_hz: float,
    basis_max_hz: float,
    basis_mode: str,
    basis_phase_count: int,
    candidate_probe_restarts: int,
    candidate_probe_scale: float,
    max_outer_iters: int,
    max_query_budget: int | None,
    spsa_steps: int,
    spsa_step_size: float,
    spsa_perturb_scale: float,
    spsa_restarts: int,
    spsa_init_scale: float,
    l2_weight: float,
    tv_weight: float,
    band_weight: float,
    max_coeff_abs: float,
    max_perturbation_peak_ratio: float | None,
    support_mode: str = "channel_window",
    channel_waveform_rank: int | None = None,
    channel_shortlist_size: int | None = None,
    enforce_unique_channels: bool = False,
    stop_on_success: bool = True,
    seed: int = 0,
    model=None,
    device=None,
):
    common_kwargs = {
        "score_fn": score_fn,
        "sfreq": sfreq,
        "n_windows": n_windows,
        "support_budget_k": support_budget_k,
        "basis_rank_r": basis_rank_r,
        "basis_min_hz": basis_min_hz,
        "basis_max_hz": basis_max_hz,
        "basis_mode": basis_mode,
        "basis_phase_count": basis_phase_count,
        "candidate_probe_restarts": candidate_probe_restarts,
        "candidate_probe_scale": candidate_probe_scale,
        "max_outer_iters": max_outer_iters,
        "max_query_budget": max_query_budget,
        "spsa_steps": spsa_steps,
        "spsa_step_size": spsa_step_size,
        "spsa_perturb_scale": spsa_perturb_scale,
        "spsa_restarts": spsa_restarts,
        "spsa_init_scale": spsa_init_scale,
        "l2_weight": l2_weight,
        "tv_weight": tv_weight,
        "band_weight": band_weight,
        "max_coeff_abs": max_coeff_abs,
        "max_perturbation_peak_ratio": max_perturbation_peak_ratio,
        "enforce_unique_channels": enforce_unique_channels,
        "stop_on_success": stop_on_success,
        "seed": seed,
    }
    if support_mode in {"saga_pgd", "saga"}:
        if model is None or device is None:
            raise ValueError("saga_pgd support_mode requires model and device")
        from .saga_attack import SagaPGDScoreAttack

        return SagaPGDScoreAttack(
            model=model,
            device=device,
            **common_kwargs,
        )
    if support_mode == "channel_window":
        return GreedySparseScoreAttack(**common_kwargs)
    if support_mode == "channel_window_freq_bank":
        return GreedySparseScoreAttack(
            **{
                **common_kwargs,
                "basis_mode": "freq_atom_bank",
            }
        )
    if support_mode == "channel_first":
        return ChannelFirstScoreAttack(
            **common_kwargs,
            channel_waveform_rank=channel_waveform_rank,
        )
    if support_mode == "qeldba":
        from .qeldba_attack import QeldbaScoreAttack

        return QeldbaScoreAttack(
            **common_kwargs,
            channel_waveform_rank=channel_waveform_rank,
        )
    if support_mode == "channel_then_window":
        return ChannelThenWindowScoreAttack(
            **common_kwargs,
            channel_waveform_rank=channel_waveform_rank,
            channel_shortlist_size=channel_shortlist_size,
        )
    raise ValueError(f"Unsupported support_mode: {support_mode}")
