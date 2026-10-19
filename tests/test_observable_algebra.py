import numpy as np

from eth_spin_chain import ObservableSpec, ObservableTerm, build_observable


def test_pauli_product_phase() -> None:
    x = ObservableSpec([ObservableTerm([0], "X")])
    y = ObservableSpec([ObservableTerm([0], "Y")])

    product = x.multiply(y)

    assert product.terms == [ObservableTerm([0], "Z", 1.0j)]


def test_commutator_of_same_site_x_and_y() -> None:
    x = ObservableSpec([ObservableTerm([0], "X")])
    y = ObservableSpec([ObservableTerm([0], "Y")])

    comm = x.commutator(y)

    assert comm.terms == [ObservableTerm([0], "Z", 2.0j)]


def test_identity_term_builds_identity_matrix() -> None:
    obs = ObservableSpec([ObservableTerm([], "", 2.0)])

    matrix = build_observable(2, obs)

    assert np.allclose(matrix, 2.0 * np.eye(4))
