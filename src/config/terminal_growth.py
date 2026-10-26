"""Post-terminal IID growth and the homogeneous Epstein--Zin continuation.

In lognormal mode, annual log(C[next]/C) is N(log_mean, log_sigma**2).
These are fresh, independent annual shocks, NOT a permanently drawn growth
regime. Only growth of the supplied terminal consumption level is replaced;
the inherited terminal climate/abatement treatment is unchanged. No pre-terminal growth
shocks or additional decision-tree branches are introduced.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os

import numpy as np

from .revised_baseline import BASELINE


class NonFiniteTerminalError(ValueError):
    """The specified infinite continuation is not economically admissible."""


def continuation_log_multiplier(beta, rho, log_growth_per_step):
    """Log K, where U_T=K*c_T, evaluated stably including EIS=1."""
    if not (np.isfinite(beta) and 0 < beta < 1):
        raise ValueError("Terminal continuation requires 0 < beta < 1")
    if not np.isfinite(rho) or not np.isfinite(log_growth_per_step):
        raise ValueError("Terminal continuation inputs must be finite")
    log_ratio = np.log(beta) + rho * log_growth_per_step
    if log_ratio >= 0:
        raise NonFiniteTerminalError(
            "Infinite terminal growing annuity: beta * "
            "terminal_growth_per_step**rho must be below one "
            f"(log ratio={log_ratio:.12g})")
    if abs(rho) <= 1e-12:
        answer = beta * log_growth_per_step / (1-beta)
    else:
        answer = -np.log1p(
            -beta/(1-beta) * np.expm1(rho*log_growth_per_step)) / rho
    if not np.isfinite(answer) or abs(answer) >= np.log(np.finfo(float).max):
        raise NonFiniteTerminalError("Terminal multiplier is outside floating-point range")
    return float(answer)


@dataclass(frozen=True)
class TerminalGrowthSettings:
    mode: str = BASELINE.terminal_growth_mode
    growth: float = BASELINE.terminal_growth
    log_mean: float = BASELINE.terminal_log_mean
    log_sigma: float = BASELINE.terminal_log_sigma

    def __post_init__(self):
        if self.mode not in ("lognormal", "deterministic", "legacy"):
            raise ValueError("Terminal growth mode must be lognormal, deterministic, or legacy")
        if not all(np.isfinite(x) for x in (self.growth, self.log_mean, self.log_sigma)):
            raise ValueError("Terminal growth parameters must be finite")
        if self.growth <= -1 or self.log_sigma < 0:
            raise ValueError("Require terminal growth > -1 and log sigma >= 0")

    @classmethod
    def from_environment(cls):
        return cls(
            mode=os.environ.get("REVISED_TERMINAL_GROWTH_MODE", BASELINE.terminal_growth_mode),
            growth=float(os.environ.get("REVISED_TERMINAL_GROWTH", BASELINE.terminal_growth)),
            log_mean=float(os.environ.get("REVISED_TERMINAL_LOG_MEAN", BASELINE.terminal_log_mean)),
            log_sigma=float(os.environ.get("REVISED_TERMINAL_LOG_SIGMA", BASELINE.terminal_log_sigma)))

    def log_growth_per_step(self, period_len, ra, modeled_growth=None):
        if not np.isfinite(period_len) or period_len <= 0:
            raise ValueError("Utility step length must be finite and positive")
        ra = np.asarray(ra, float)
        if np.any(~np.isfinite(ra)) or np.any(ra <= 0):
            raise ValueError("Require finite positive risk aversion")
        if self.mode == "lognormal":
            # E[G^(1-RA)]^(1/(1-RA)) for a compounded annual lognormal G.
            return period_len * (self.log_mean + 0.5*(1-ra)*self.log_sigma**2)
        if self.mode == "deterministic":
            return period_len * np.log1p(self.growth) + np.zeros_like(ra)
        # Explicit historical reproduction of the inherited timing error.
        growth = self.growth if modeled_growth is None else np.asarray(modeled_growth, float)
        if np.any(~np.isfinite(growth)) or np.any(growth <= -1):
            raise ValueError("Require finite modeled growth > -1")
        return np.log1p(growth) + np.zeros_like(ra)

    def admissible_mask(self, ra, eis, prtp, period_len=5, modeled_growth=None):
        """Check joint preference/growth admissibility without exponentiating."""
        ra, eis, prtp = np.broadcast_arrays(ra, eis, prtp)
        valid = (np.isfinite(ra) & np.isfinite(eis) & np.isfinite(prtp)
                 & (ra > 0) & (eis > 0) & (prtp > 0) & (prtp < 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = (period_len*np.log1p(-prtp)
                + (1-1/eis)*self.log_growth_per_step(period_len, ra, modeled_growth))
        return valid & (log_ratio < 0)

    def describe(self, *, ra, eis, prtp, period_len=5, modeled_growth=None):
        if not (np.isfinite(eis) and eis > 0 and np.isfinite(prtp) and 0 < prtp < 1):
            raise ValueError("Require EIS > 0 and 0 < PRTP < 1")
        log_step = float(self.log_growth_per_step(period_len, ra, modeled_growth))
        beta = (1-prtp)**period_len
        rho = 1-1/eis
        log_k = continuation_log_multiplier(beta, rho, log_step)
        return {**asdict(self), "utility_step_years": float(period_len),
                "shock_process": ("IID annual lognormal gross consumption growth"
                                  if self.mode == "lognormal" else None),
                "effective_annual_log_growth": log_step/period_len,
                "effective_annual_net_growth": float(np.expm1(log_step/period_len)),
                "expected_annual_net_growth": (float(np.expm1(self.log_mean+0.5*self.log_sigma**2))
                    if self.mode == "lognormal" else self.growth),
                "growth_factor_per_utility_step": float(np.exp(log_step)),
                "continuation_ratio": float(beta*np.exp(rho*log_step)),
                "multiplier": float(np.exp(log_k)),
                "climate_and_cost_tail": "terminal consumption scaled by growth; no new climate dynamics or controls"}

    def model_kwargs(self):
        return dict(terminal_growth_mode=self.mode, terminal_growth=self.growth,
                    terminal_log_mean=self.log_mean, terminal_log_sigma=self.log_sigma)

    def cli_args(self):
        return ["--terminal-growth-mode", self.mode, "--terminal-growth", str(self.growth),
                "--terminal-log-mean", str(self.log_mean), "--terminal-log-sigma", str(self.log_sigma)]

    def cache_tag(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return "terminal_" + hashlib.sha256(payload).hexdigest()[:12]


def add_terminal_arguments(parser):
    """Identical terminal settings for the CLI and every cluster wrapper."""
    settings = TerminalGrowthSettings.from_environment()
    parser.add_argument("--terminal-growth-mode", choices=("lognormal", "deterministic", "legacy"),
                        default=settings.mode, help="Post-terminal consumption growth process")
    parser.add_argument("--terminal-growth", type=float, default=settings.growth,
                        help="Annual NET growth after the terminal date in deterministic mode")
    parser.add_argument("--terminal-log-mean", type=float, default=settings.log_mean,
                        help="Annual mean LOG gross growth after the terminal date")
    parser.add_argument("--terminal-log-sigma", type=float, default=settings.log_sigma,
                        help="Annual standard deviation of LOG gross growth after the terminal date")


def terminal_settings_from_args(args):
    return TerminalGrowthSettings(args.terminal_growth_mode, args.terminal_growth,
                                  args.terminal_log_mean, args.terminal_log_sigma)
