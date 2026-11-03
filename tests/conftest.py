"""Shared pytest fixtures. Everything here is synthetic: no robot, no GPU."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
