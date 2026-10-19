from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eth_spin_chain import (
    DiagonalizedHamiltonian,
    EnergyWindow,
    HamiltonianSpec,
    InteractionTerm,
    ObservableSpec,
    ObservableTerm,
    eth_elements,
)
from standard_models import transverse_field_ising_chain


def test_eth_elements_returns_full_spectrum_and_diagonal() -> None:
    ham = transverse_field_ising_chain(L=3, h=0.7, g=0.2)
    obs = ObservableSpec([ObservableTerm([0], "Z")])

    result = eth_elements(ham, obs)

    assert result.energies.shape == (8,)
    assert result.diagonal.shape == (8,)
    assert np.all(np.diff(result.energies) >= -1e-12)


def test_eth_window_statistics_are_populated() -> None:
    ham = transverse_field_ising_chain(L=3, h=0.7, g=0.2)
    obs = ObservableSpec([ObservableTerm([0], "Z")])
    diagonalized = DiagonalizedHamiltonian.from_spec(ham)

    result = eth_elements(
        diagonalized,
        obs,
        window=EnergyWindow(center=0.0, width=10.0),
        max_window_dim=16,
    )

    assert result.window_indices is not None
    assert result.offdiag_abs_mean is not None
    assert result.offdiag_abs_std is not None


def test_eth_window_statistics_exclude_diagonal_from_denominator() -> None:
    spec = transverse_field_ising_chain(L=2, j=0.0, h=0.0, g=0.0)
    diagonalized = DiagonalizedHamiltonian.from_spec(spec)
    obs = ObservableSpec([ObservableTerm([0], "X")])

    result = eth_elements(
        diagonalized,
        obs,
        window=EnergyWindow(center=0.0, width=1.0),
        return_full_window_matrix=True,
        max_window_dim=4,
    )

    window_mat = result.window_matrix
    assert window_mat is not None
    offdiag_mask = ~np.eye(window_mat.shape[0], dtype=bool)
    expected = np.abs(window_mat[offdiag_mask])

    assert result.offdiag_abs_mean == np.mean(expected)
    assert result.offdiag_abs_std == np.std(expected)


def test_eth_window_statistics_require_at_least_two_states() -> None:
    spec = HamiltonianSpec(
        L=2,
        terms=[
            InteractionTerm([0], "Z", coeff=1.0),
            InteractionTerm([1], "Z", coeff=0.37),
        ],
    )
    diagonalized = DiagonalizedHamiltonian.from_spec(spec)
    obs = ObservableSpec([ObservableTerm([0], "X")])

    with pytest.raises(ValueError, match="At least two states"):
        eth_elements(
            diagonalized,
            obs,
            window=EnergyWindow(center=-1.37, width=0.1),
        )
