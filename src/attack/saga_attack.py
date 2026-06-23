from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .basis import build_basis_matrix
from .greedy_attack import AttackResult, GreedySparseScoreAttack, QueryBudgetExhausted
from .losses import untargeted_margin
from .support import all_atoms, make_window_partition


class SagaPGDScoreAttack(GreedySparseScoreAttack):
    def __init__(self, model, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _count_query(self, n: int = 1) -> None:
        if self.max_query_budget is not None and self.queries_used + n > self.max_query_budget:
            raise QueryBudgetExhausted(
                f"Query budget exhausted at {self.queries_used} / {self.max_query_budget} queries."
            )
        self.queries_used += n

    def _build_basis_cache(self, n_samples: int):
        partition = make_window_partition(n_samples=n_samples, n_windows=self.n_windows)
        basis_by_window_np: dict[int, np.ndarray] = {}
        basis_by_window_torch: dict[int, torch.Tensor] = {}
        for window, (start, end) in enumerate(partition.boundaries):
            basis_np = build_basis_matrix(
                basis_mode=self.basis_mode,
                window_length=end - start,
                rank=self.basis_rank_r,
                f_min_hz=self.basis_min_hz,
                f_max_hz=self.basis_max_hz,
                sfreq=self.sfreq,
                phase_count=self.basis_phase_count,
            )
            basis_by_window_np[window] = basis_np
            basis_by_window_torch[window] = torch.as_tensor(basis_np, dtype=torch.float32, device=self.device)
        return partition, basis_by_window_np, basis_by_window_torch

    def _assemble_delta_torch(
        self,
        n_channels: int,
        n_samples: int,
        partition,
        basis_by_window_torch: dict[int, torch.Tensor],
        support: list[tuple[int, int]],
        coeffs: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.zeros((n_channels, n_samples), dtype=torch.float32, device=self.device)
        if len(support) == 0:
            return delta

        for (channel, window), atom_coeffs in zip(support, coeffs):
            start, end = partition.boundaries[window]
            local = atom_coeffs @ basis_by_window_torch[window]
            channel_mask = torch.zeros((n_channels, 1), dtype=torch.float32, device=self.device)
            channel_mask[channel, 0] = 1.0
            local_full = torch.cat(
                [
                    torch.zeros((1, start), dtype=torch.float32, device=self.device),
                    local.unsqueeze(0),
                    torch.zeros((1, n_samples - end), dtype=torch.float32, device=self.device),
                ],
                dim=1,
            )
            delta = delta + channel_mask * local_full
        return delta

    def _apply_peak_ratio_constraint_torch(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.max_perturbation_peak_ratio is None:
            return delta
        signal_peak = float(torch.max(torch.abs(x)).detach().cpu().item())
        if signal_peak <= 0.0:
            return delta
        allowed_peak = float(self.max_perturbation_peak_ratio) * signal_peak
        delta_peak = float(torch.max(torch.abs(delta)).detach().cpu().item())
        if delta_peak <= allowed_peak or delta_peak <= 0.0:
            return delta
        return delta * (allowed_peak / delta_peak)

    def _evaluate_margin_torch(
        self,
        x_tensor: torch.Tensor,
        y_int: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition,
        basis_by_window_np: dict[int, np.ndarray],
        basis_by_window_torch: dict[int, torch.Tensor],
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        self._count_query()
        with torch.no_grad():
            coeffs_tensor = torch.as_tensor(coeffs, dtype=torch.float32, device=self.device)
            delta = self._assemble_delta_torch(
                n_channels=int(x_tensor.shape[1]),
                n_samples=int(x_tensor.shape[2]),
                partition=partition,
                basis_by_window_torch=basis_by_window_torch,
                support=support,
                coeffs=coeffs_tensor,
            )
            delta = self._apply_peak_ratio_constraint_torch(x_tensor, delta)
            logits = self.model(x_tensor + delta.unsqueeze(0))
        logits_np = logits.detach().cpu().numpy().squeeze(0)
        return untargeted_margin(logits_np, y_int), logits, delta

    def _input_gradient(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition,
        basis_by_window_torch: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        self._count_query()
        coeffs_tensor = torch.as_tensor(coeffs, dtype=torch.float32, device=self.device)
        delta = self._assemble_delta_torch(
            n_channels=int(x_tensor.shape[1]),
            n_samples=int(x_tensor.shape[2]),
            partition=partition,
            basis_by_window_torch=basis_by_window_torch,
            support=support,
            coeffs=coeffs_tensor,
        )
        delta = self._apply_peak_ratio_constraint_torch(x_tensor, delta)
        x_adv = (x_tensor + delta.unsqueeze(0)).detach().requires_grad_(True)
        logits = self.model(x_adv)
        loss = F.cross_entropy(logits, y_tensor)
        grad = torch.autograd.grad(loss, x_adv)[0]
        return grad.detach()

    def _project_gradient_to_coeffs(
        self,
        grad: torch.Tensor,
        atom: tuple[int, int],
        partition,
        basis_by_window_np: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        channel, window = atom
        start, end = partition.boundaries[window]
        window_grad = grad[0, channel, start:end].detach().cpu().numpy().astype(np.float32)
        basis = basis_by_window_np[window].astype(np.float32, copy=False)
        coeffs = window_grad @ basis.T
        coeff_norm = float(np.linalg.norm(coeffs))
        if coeff_norm > 0.0:
            coeffs = coeffs / coeff_norm
        coeffs = coeffs * float(self.spsa_init_scale) * float(self.max_coeff_abs)
        if self.spsa_perturb_scale > 0.0:
            coeffs = coeffs + float(self.spsa_perturb_scale) * float(self.max_coeff_abs) * rng.choice(
                [-1.0, 1.0], size=coeffs.shape
            ).astype(np.float32)
        return np.clip(coeffs, -self.max_coeff_abs, self.max_coeff_abs).astype(np.float32, copy=False)

    def _probe_initial_coeffs(
        self,
        x_tensor: torch.Tensor,
        y_int: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition,
        basis_by_window_np: dict[int, np.ndarray],
        basis_by_window_torch: dict[int, torch.Tensor],
        rng: np.random.Generator,
    ) -> np.ndarray:
        best_coeffs = coeffs.copy()
        best_margin, _, _ = self._evaluate_margin_torch(
            x_tensor=x_tensor,
            y_int=y_int,
            support=support,
            coeffs=best_coeffs,
            partition=partition,
            basis_by_window_np=basis_by_window_np,
            basis_by_window_torch=basis_by_window_torch,
        )
        if self.candidate_probe_restarts <= 0:
            return best_coeffs

        probe_scale = float(self.candidate_probe_scale) * float(self.max_coeff_abs)
        for _ in range(self.candidate_probe_restarts):
            proposal = coeffs.copy()
            if probe_scale > 0.0:
                proposal = proposal + probe_scale * rng.choice([-1.0, 1.0], size=proposal.shape).astype(np.float32)
            proposal = np.clip(proposal, -self.max_coeff_abs, self.max_coeff_abs).astype(np.float32, copy=False)
            margin, _, _ = self._evaluate_margin_torch(
                x_tensor=x_tensor,
                y_int=y_int,
                support=support,
                coeffs=proposal,
                partition=partition,
                basis_by_window_np=basis_by_window_np,
                basis_by_window_torch=basis_by_window_torch,
            )
            if margin < best_margin:
                best_margin = margin
                best_coeffs = proposal
        return best_coeffs

    def _refine_coeffs(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        y_int: int,
        support: list[tuple[int, int]],
        coeffs: np.ndarray,
        partition,
        basis_by_window_np: dict[int, np.ndarray],
        basis_by_window_torch: dict[int, torch.Tensor],
    ) -> tuple[np.ndarray, float]:
        current = coeffs.copy()
        best_coeffs = current.copy()
        best_margin, _, _ = self._evaluate_margin_torch(
            x_tensor=x_tensor,
            y_int=y_int,
            support=support,
            coeffs=current,
            partition=partition,
            basis_by_window_np=basis_by_window_np,
            basis_by_window_torch=basis_by_window_torch,
        )

        total_steps = max(1, int(self.spsa_steps))
        for _ in range(total_steps):
            coeffs_tensor = torch.as_tensor(current, dtype=torch.float32, device=self.device).detach().requires_grad_(True)
            self._count_query()
            delta = self._assemble_delta_torch(
                n_channels=int(x_tensor.shape[1]),
                n_samples=int(x_tensor.shape[2]),
                partition=partition,
                basis_by_window_torch=basis_by_window_torch,
                support=support,
                coeffs=coeffs_tensor,
            )
            delta = self._apply_peak_ratio_constraint_torch(x_tensor, delta)
            logits = self.model(x_tensor + delta.unsqueeze(0))
            ce = F.cross_entropy(logits, y_tensor)
            l2 = torch.mean(coeffs_tensor**2)
            if delta.shape[-1] <= 1:
                tv = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                tv = torch.mean(torch.abs(delta[:, 1:] - delta[:, :-1]))
            if self.band_weight == 0.0:
                band = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                fft = torch.fft.rfft(delta, dim=-1)
                freqs = torch.fft.rfftfreq(delta.shape[-1], d=1.0 / self.sfreq).to(self.device)
                high_mask = freqs > 40.0
                if torch.any(high_mask):
                    band = torch.mean(torch.abs(fft[:, high_mask]) ** 2)
                else:
                    band = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            objective = -ce + self.l2_weight * l2 + self.tv_weight * tv + self.band_weight * band
            grad = torch.autograd.grad(objective, coeffs_tensor)[0].detach().cpu().numpy().astype(np.float32)

            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 0.0:
                current = np.clip(current - self.spsa_step_size * (grad / grad_norm), -self.max_coeff_abs, self.max_coeff_abs).astype(np.float32, copy=False)

            current_margin, _, _ = self._evaluate_margin_torch(
                x_tensor=x_tensor,
                y_int=y_int,
                support=support,
                coeffs=current,
                partition=partition,
                basis_by_window_np=basis_by_window_np,
                basis_by_window_torch=basis_by_window_torch,
            )
            if current_margin < best_margin:
                best_margin = current_margin
                best_coeffs = current.copy()
            if best_margin < 0.0 and self.stop_on_success:
                return best_coeffs, best_margin

        final_margin, _, _ = self._evaluate_margin_torch(
            x_tensor=x_tensor,
            y_int=y_int,
            support=support,
            coeffs=current,
            partition=partition,
            basis_by_window_np=basis_by_window_np,
            basis_by_window_torch=basis_by_window_torch,
        )
        if final_margin < best_margin:
            best_margin = final_margin
            best_coeffs = current.copy()
        return best_coeffs, best_margin

    def _choose_atom(
        self,
        grad: torch.Tensor,
        support: list[tuple[int, int]],
        partition,
        selected_atoms: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        best_atom: tuple[int, int] | None = None
        best_score = float("-inf")
        for atom in all_atoms(int(grad.shape[1]), partition):
            if atom in selected_atoms:
                continue
            if self.enforce_unique_channels and any(int(existing_channel) == int(atom[0]) for existing_channel, _ in support):
                continue
            channel, window = atom
            start, end = partition.boundaries[window]
            score = float(torch.mean(torch.abs(grad[0, channel, start:end])).detach().cpu().item())
            if score > best_score:
                best_score = score
                best_atom = atom
        return best_atom

    def _run_single_pass(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        y_int: int,
        partition,
        basis_by_window_np: dict[int, np.ndarray],
        basis_by_window_torch: dict[int, torch.Tensor],
        rng: np.random.Generator,
    ) -> AttackResult:
        n_channels = int(x_tensor.shape[1])
        n_samples = int(x_tensor.shape[2])
        support: list[tuple[int, int]] = []
        coeffs = np.zeros((0, self.basis_rank_r), dtype=np.float32)
        selected_atoms: set[tuple[int, int]] = set()

        initial_margin, _, _ = self._evaluate_margin_torch(
            x_tensor=x_tensor,
            y_int=y_int,
            support=support,
            coeffs=coeffs,
            partition=partition,
            basis_by_window_np=basis_by_window_np,
            basis_by_window_torch=basis_by_window_torch,
        )

        if initial_margin < 0.0:
            return self._build_result(
                x=x_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32),
                y=y_int,
                partition=partition,
                basis_by_window=basis_by_window_np,
                support=support,
                coeffs=coeffs,
                margin=initial_margin,
                budget_exhausted=False,
            )

        current_margin = initial_margin
        try:
            outer_steps = min(int(self.support_budget_k), int(self.max_outer_iters))
            for _ in range(outer_steps):
                grad = self._input_gradient(
                    x_tensor=x_tensor,
                    y_tensor=y_tensor,
                    support=support,
                    coeffs=coeffs,
                    partition=partition,
                    basis_by_window_torch=basis_by_window_torch,
                )
                atom = self._choose_atom(
                    grad=grad,
                    support=support,
                    partition=partition,
                    selected_atoms=selected_atoms,
                )
                if atom is None:
                    break

                selected_atoms.add(atom)
                support.append(atom)
                init_coeffs = self._project_gradient_to_coeffs(
                    grad=grad,
                    atom=atom,
                    partition=partition,
                    basis_by_window_np=basis_by_window_np,
                    rng=rng,
                )
                coeffs = np.vstack([coeffs, init_coeffs[None, :]]).astype(np.float32, copy=False)
                coeffs = self._probe_initial_coeffs(
                    x_tensor=x_tensor,
                    y_int=y_int,
                    support=support,
                    coeffs=coeffs,
                    partition=partition,
                    basis_by_window_np=basis_by_window_np,
                    basis_by_window_torch=basis_by_window_torch,
                    rng=rng,
                )
                coeffs, current_margin = self._refine_coeffs(
                    x_tensor=x_tensor,
                    y_tensor=y_tensor,
                    y_int=y_int,
                    support=support,
                    coeffs=coeffs,
                    partition=partition,
                    basis_by_window_np=basis_by_window_np,
                    basis_by_window_torch=basis_by_window_torch,
                )
                if current_margin < 0.0 and self.stop_on_success:
                    break
        except QueryBudgetExhausted:
            return self._build_result(
                x=x_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32),
                y=y_int,
                partition=partition,
                basis_by_window=basis_by_window_np,
                support=support,
                coeffs=coeffs,
                margin=current_margin,
                budget_exhausted=True,
            )

        return self._build_result(
            x=x_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32),
            y=y_int,
            partition=partition,
            basis_by_window=basis_by_window_np,
            support=support,
            coeffs=coeffs,
            margin=current_margin,
            budget_exhausted=False,
        )

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        self.queries_used = 0
        x_tensor = torch.as_tensor(x[None], dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor([int(y)], dtype=torch.long, device=self.device)
        partition, basis_by_window_np, basis_by_window_torch = self._build_basis_cache(x.shape[1])

        best_result: AttackResult | None = None
        restarts = max(1, int(self.spsa_restarts))
        for restart in range(restarts):
            rng = np.random.default_rng(self.seed + restart)
            result = self._run_single_pass(
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                y_int=int(y),
                partition=partition,
                basis_by_window_np=basis_by_window_np,
                basis_by_window_torch=basis_by_window_torch,
                rng=rng,
            )
            if best_result is None:
                best_result = result
            elif result.success and not best_result.success:
                best_result = result
            elif result.success == best_result.success and result.margin < best_result.margin:
                best_result = result
            if result.success and self.stop_on_success:
                break

        if best_result is None:
            raise RuntimeError("SAGA attack failed to produce a result")
        return best_result