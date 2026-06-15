import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from main import Diffusion


def test_sample_timesteps_biased_high_t_fraction():
    diff = Diffusion(noise_steps=1000)
    draws = diff.sample_timesteps_biased(10000, high_prob=0.5, high_threshold=700)
    high_frac = (draws >= 700).float().mean().item()
    low_frac = ((draws >= 1) & (draws < 700)).float().mean().item()
    assert 0.45 <= high_frac <= 0.55, f"high-t fraction {high_frac}"
    assert 0.45 <= low_frac <= 0.55, f"low-t fraction {low_frac}"


def test_sample_timesteps_biased_respects_threshold():
    diff = Diffusion(noise_steps=1000)
    draws = diff.sample_timesteps_biased(1000, high_prob=1.0, high_threshold=700)
    assert draws.min().item() >= 700
    assert draws.max().item() < 1000

    draws_low = diff.sample_timesteps_biased(1000, high_prob=0.0, high_threshold=700)
    assert draws_low.min().item() >= 1
    assert draws_low.max().item() < 700
