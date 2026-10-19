from __future__ import annotations

from itertools import product
from numbers import Real

import numpy as np

from eth_spin_chain import (
    Boundary,
    HamiltonianSpec,
    InteractionTerm,
    sample_couplings,
    sample_interaction_terms,
)


RangeLike = float | tuple[float, float]


def _as_range(value: RangeLike, name: str) -> tuple[float, float]:
    if isinstance(value, Real):
        scalar = float(value)
        return scalar, scalar
    if len(value) != 2:
        raise ValueError(f"{name} must be a float or a (min, max) tuple")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{name} range must satisfy min <= max")
    return low, high


def transverse_field_ising_chain(
    L: int,
    j: float = 1.0,
    h: float = 1.0,
    g: float = 0.0,
    boundary: Boundary = "periodic",
) -> HamiltonianSpec:
    """Return -j sum ZZ - h sum X - g sum Z."""
    terms: list[InteractionTerm] = []
    terms.extend(sample_interaction_terms(L, "ZZ", j, j, boundary=boundary))
    terms = [InteractionTerm(term.sites, term.ops, -term.coeff) for term in terms]
    for site in range(L):
        terms.append(InteractionTerm(sites=[site], ops="X", coeff=-h))
        if g != 0.0:
            terms.append(InteractionTerm(sites=[site], ops="Z", coeff=-g))
    return HamiltonianSpec(L=L, terms=terms, boundary=boundary)


def xxz_chain(
    L: int,
    jxy: float = 1.0,
    jz: float = 1.0,
    boundary: Boundary = "periodic",
) -> HamiltonianSpec:
    """Return sum jxy(XX + YY) + jz ZZ."""
    terms: list[InteractionTerm] = []
    terms.extend(sample_interaction_terms(L, "XX", jxy, jxy, boundary=boundary))
    terms.extend(sample_interaction_terms(L, "YY", jxy, jxy, boundary=boundary))
    terms.extend(sample_interaction_terms(L, "ZZ", jz, jz, boundary=boundary))
    return HamiltonianSpec(L=L, terms=terms, boundary=boundary)


def disordered_heisenberg_chain(
    L: int,
    jx: RangeLike = 1.0,
    jy: RangeLike = 1.0,
    jz: RangeLike = 1.0,
    hx: RangeLike = 0.0,
    hy: RangeLike = 0.0,
    hz: RangeLike | None = None,
    field_strength: float = 0.5,
    boundary: Boundary = "periodic",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> HamiltonianSpec:
    """Return anisotropic Heisenberg chain with configurable weak disorder.

    Interaction disorder can be specified independently for XX, YY, and ZZ via
    either a fixed float or a (min, max) range. Onsite fields can be specified
    similarly for X, Y, and Z.

    Backward compatibility: if ``hz`` is not provided, the Z field defaults to
    a symmetric range ``(-field_strength, field_strength)``.
    """
    if rng is not None and seed is not None:
        raise ValueError("rng and seed cannot both be provided")

    generator = np.random.default_rng(seed) if rng is None else rng

    jx_min, jx_max = _as_range(jx, "jx")
    jy_min, jy_max = _as_range(jy, "jy")
    jz_min, jz_max = _as_range(jz, "jz")
    hx_min, hx_max = _as_range(hx, "hx")
    hy_min, hy_max = _as_range(hy, "hy")
    if hz is None:
        hz_min, hz_max = -float(field_strength), float(field_strength)
    else:
        hz_min, hz_max = _as_range(hz, "hz")

    terms: list[InteractionTerm] = []
    terms.extend(
        sample_interaction_terms(
            L,
            "XX",
            jx_min,
            jx_max,
            boundary=boundary,
            rng=generator,
        )
    )
    terms.extend(
        sample_interaction_terms(
            L,
            "YY",
            jy_min,
            jy_max,
            boundary=boundary,
            rng=generator,
        )
    )
    terms.extend(
        sample_interaction_terms(
            L,
            "ZZ",
            jz_min,
            jz_max,
            boundary=boundary,
            rng=generator,
        )
    )

    hx_fields = sample_couplings(L, hx_min, hx_max, rng=generator)
    hy_fields = sample_couplings(L, hy_min, hy_max, rng=generator)
    hz_fields = sample_couplings(L, hz_min, hz_max, rng=generator)

    for site in range(L):
        if hx_fields[site] != 0.0:
            terms.append(InteractionTerm(sites=[site], ops="X", coeff=float(hx_fields[site])))
        if hy_fields[site] != 0.0:
            terms.append(InteractionTerm(sites=[site], ops="Y", coeff=float(hy_fields[site])))
        if hz_fields[site] != 0.0:
            terms.append(InteractionTerm(sites=[site], ops="Z", coeff=float(hz_fields[site])))
    return HamiltonianSpec(L=L, terms=terms, boundary=boundary)


def z2z2_projective_xyz_chain(
    L: int,
    jx: RangeLike = 1.0,
    jy: RangeLike = 1.0,
    jz: RangeLike = 1.0,
    alpha: float = 0.0,
    boundary: Boundary = "periodic",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> HamiltonianSpec:
    """Return the Eq. (5.14)-style Z2 x Z2 spin chain Hamiltonian.

    The model is

        H = sum_j (Jx_j X_j X_{j+1} + Jy_j Y_j Y_{j+1} + Jz_j Z_j Z_{j+1})
            + alpha * sum_j X_j Y_{j+1} Z_{j+2}

    where the two-body couplings can be fixed or weakly disordered per channel.
    """
    if rng is not None and seed is not None:
        raise ValueError("rng and seed cannot both be provided")

    generator = np.random.default_rng(seed) if rng is None else rng

    jx_min, jx_max = _as_range(jx, "jx")
    jy_min, jy_max = _as_range(jy, "jy")
    jz_min, jz_max = _as_range(jz, "jz")

    terms: list[InteractionTerm] = []
    terms.extend(
        sample_interaction_terms(
            L,
            "XX",
            jx_min,
            jx_max,
            boundary=boundary,
            rng=generator,
        )
    )
    terms.extend(
        sample_interaction_terms(
            L,
            "YY",
            jy_min,
            jy_max,
            boundary=boundary,
            rng=generator,
        )
    )
    terms.extend(
        sample_interaction_terms(
            L,
            "ZZ",
            jz_min,
            jz_max,
            boundary=boundary,
            rng=generator,
        )
    )

    if alpha != 0.0:
        terms.extend(
            sample_interaction_terms(
                L,
                "XYZ",
                float(alpha),
                float(alpha),
                boundary=boundary,
                rng=generator,
            )
        )

    return HamiltonianSpec(L=L, terms=terms, boundary=boundary)


def random_pauli_hamiltonian(
    L: int,
    term_width: int = 2,
    n_terms: int | None = None,
    coeff_min: float = -1.0,
    coeff_max: float = 1.0,
    seed: int | None = None,
) -> HamiltonianSpec:
    """Return a reproducible open-chain random Pauli-string Hamiltonian."""
    if L < 2:
        raise ValueError("L must be >= 2")
    if not 1 <= term_width <= L:
        raise ValueError("term_width must satisfy 1 <= term_width <= L")

    all_terms = [
        (start, "".join(ops))
        for start in range(L - term_width + 1)
        for ops in product("XYZ", repeat=term_width)
    ]
    if n_terms is None:
        n_terms = min(len(all_terms), 2 * L)
    if not 1 <= n_terms <= len(all_terms):
        raise ValueError(f"n_terms must satisfy 1 <= n_terms <= {len(all_terms)}")

    generator = np.random.default_rng(seed)
    choices = generator.choice(len(all_terms), size=n_terms, replace=False)
    coeffs = sample_couplings(n_terms, coeff_min, coeff_max, rng=generator)

    terms: list[InteractionTerm] = []
    for choice, coeff in zip(choices, coeffs):
        start, ops = all_terms[int(choice)]
        sites = list(range(start, start + term_width))
        terms.append(InteractionTerm(sites=sites, ops=ops, coeff=float(coeff)))
    return HamiltonianSpec(L=L, terms=terms, boundary="open")
