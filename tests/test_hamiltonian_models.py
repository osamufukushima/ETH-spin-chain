from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eth_spin_chain import HamiltonianSpec, InteractionTerm, build_hamiltonian, sample_interaction_terms
from standard_models import disordered_heisenberg_chain, xxz_chain, z2z2_projective_xyz_chain


def test_single_site_z_hamiltonian() -> None:
    spec = HamiltonianSpec(L=2, terms=[InteractionTerm([0], "Z")])

    matrix = build_hamiltonian(spec)

    expected = np.diag([1, 1, -1, -1])
    assert np.allclose(matrix, expected)


def test_sample_open_terms_do_not_wrap() -> None:
    terms = sample_interaction_terms(L=4, ops="XX", j_min=1.0, j_max=1.0, boundary="open")

    assert [term.sites for term in terms] == [[0, 1], [1, 2], [2, 3]]


def test_xxz_chain_has_three_terms_per_periodic_bond() -> None:
    spec = xxz_chain(L=4)

    assert len(spec.terms) == 12


def test_disordered_heisenberg_supports_full_weak_disorder_ranges() -> None:
    spec = disordered_heisenberg_chain(
        L=4,
        jx=(0.9, 1.1),
        jy=(0.8, 1.2),
        jz=(0.7, 1.3),
        hx=(-0.05, 0.05),
        hy=(-0.04, 0.04),
        hz=(-0.03, 0.03),
        seed=123,
    )

    xx_terms = [term for term in spec.terms if term.ops == "XX"]
    yy_terms = [term for term in spec.terms if term.ops == "YY"]
    zz_terms = [term for term in spec.terms if term.ops == "ZZ"]
    x_terms = [term for term in spec.terms if term.ops == "X"]
    y_terms = [term for term in spec.terms if term.ops == "Y"]
    z_terms = [term for term in spec.terms if term.ops == "Z" and len(term.sites) == 1]

    assert len(xx_terms) == 4
    assert len(yy_terms) == 4
    assert len(zz_terms) == 4
    assert len(x_terms) == 4
    assert len(y_terms) == 4
    assert len(z_terms) == 4

    assert all(0.9 <= term.coeff <= 1.1 for term in xx_terms)
    assert all(0.8 <= term.coeff <= 1.2 for term in yy_terms)
    assert all(0.7 <= term.coeff <= 1.3 for term in zz_terms)
    assert all(-0.05 <= term.coeff <= 0.05 for term in x_terms)
    assert all(-0.04 <= term.coeff <= 0.04 for term in y_terms)
    assert all(-0.03 <= term.coeff <= 0.03 for term in z_terms)


def test_disordered_heisenberg_old_field_strength_path_still_works() -> None:
    spec = disordered_heisenberg_chain(
        L=4,
        jx=1.0,
        jy=1.0,
        jz=1.0,
        field_strength=0.2,
        seed=7,
    )

    z_fields = [term for term in spec.terms if term.ops == "Z" and len(term.sites) == 1]
    assert len(z_fields) == 4
    assert all(-0.2 <= term.coeff <= 0.2 for term in z_fields)


def test_z2z2_projective_xyz_chain_periodic_term_counts_and_ranges() -> None:
    spec = z2z2_projective_xyz_chain(
        L=5,
        jx=(0.9, 1.0),
        jy=(0.7, 0.8),
        jz=(0.6, 0.7),
        alpha=0.9,
        seed=11,
    )

    xx_terms = [term for term in spec.terms if term.ops == "XX"]
    yy_terms = [term for term in spec.terms if term.ops == "YY"]
    zz_terms = [term for term in spec.terms if term.ops == "ZZ"]
    xyz_terms = [term for term in spec.terms if term.ops == "XYZ"]

    assert len(xx_terms) == 5
    assert len(yy_terms) == 5
    assert len(zz_terms) == 5
    assert len(xyz_terms) == 5

    assert all(0.9 <= term.coeff <= 1.0 for term in xx_terms)
    assert all(0.7 <= term.coeff <= 0.8 for term in yy_terms)
    assert all(0.6 <= term.coeff <= 0.7 for term in zz_terms)
    assert all(term.coeff == 0.9 for term in xyz_terms)


def test_z2z2_projective_xyz_chain_open_boundary_counts() -> None:
    spec = z2z2_projective_xyz_chain(
        L=5,
        jx=1.0,
        jy=1.0,
        jz=1.0,
        alpha=0.5,
        boundary="open",
        seed=3,
    )

    xx_terms = [term for term in spec.terms if term.ops == "XX"]
    xyz_terms = [term for term in spec.terms if term.ops == "XYZ"]

    assert len(xx_terms) == 4
    assert len(xyz_terms) == 3


def test_z2z2_projective_xyz_chain_alpha_zero_skips_three_body_terms() -> None:
    spec = z2z2_projective_xyz_chain(L=6, alpha=0.0, seed=5)
    xyz_terms = [term for term in spec.terms if term.ops == "XYZ"]
    assert len(xyz_terms) == 0
