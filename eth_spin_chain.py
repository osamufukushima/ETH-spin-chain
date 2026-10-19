from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
from scipy.sparse import csr_matrix, eye, kron


Boundary = Literal["open", "periodic"]


PAULI = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _validate_sites_ops(sites: Sequence[int], ops: str, allow_empty: bool = False) -> None:
    if not allow_empty and len(sites) == 0:
        raise ValueError("sites must not be empty")
    if len(sites) != len(ops):
        raise ValueError("len(sites) must match len(ops)")
    allowed = set(PAULI.keys())
    if any(ch not in allowed for ch in ops):
        raise ValueError("ops can only contain I, X, Y, Z")


################################################################################################################################
################################################################################################################################
# Data classes for Hamiltonian and observable specifications, energy windows, and ETH results


## Observable term specification
@dataclass(frozen=True)
class ObservableTerm:
    sites: Sequence[int]
    ops: str
    coeff: float | complex = 1.0

    # Allows an empty sequence to represent the identity operator.
    def __post_init__(self) -> None:
        _validate_sites_ops(self.sites, self.ops, allow_empty=True)


## List of observable terms
@dataclass(frozen=True)
class ObservableSpec:
    terms: Sequence[ObservableTerm]

    def __post_init__(self) -> None:
        # Empty terms represent the zero operator.
        for term in self.terms:
            if len(term.sites) != len(term.ops):
                raise ValueError("each term must satisfy len(sites) == len(ops)")

    def multiply(self, other: "ObservableSpec", tol: float = 1e-12) -> "ObservableSpec":
        if tol < 0.0:
            raise ValueError("tol must be >= 0")

        accum: dict[tuple[tuple[int, ...], str], complex] = defaultdict(complex)
        for lhs in self.terms:
            for rhs in other.terms:
                coeff, sites, ops = _multiply_observable_terms(lhs, rhs)
                key = (sites, ops)
                accum[key] += coeff

        out_terms: list[ObservableTerm] = []
        for (sites, ops), coeff in accum.items():
            if abs(coeff) <= tol:
                continue
            # Keep real coefficients as float for cleaner downstream behavior.
            coeff_out: float | complex
            if abs(coeff.imag) <= tol:
                coeff_out = float(coeff.real)
            else:
                coeff_out = coeff
            out_terms.append(ObservableTerm(sites=list(sites), ops=ops, coeff=coeff_out))

        out_terms.sort(key=lambda t: (len(t.sites), tuple(t.sites), t.ops))
        return ObservableSpec(terms=out_terms)

    def commutator(self, other: "ObservableSpec", tol: float = 1e-12) -> "ObservableSpec":
        if tol < 0.0:
            raise ValueError("tol must be >= 0")

        ab = self.multiply(other, tol=tol)
        ba = other.multiply(self, tol=tol)

        accum: dict[tuple[tuple[int, ...], str], complex] = defaultdict(complex)
        for term in ab.terms:
            accum[(tuple(term.sites), term.ops)] += complex(term.coeff)
        for term in ba.terms:
            accum[(tuple(term.sites), term.ops)] -= complex(term.coeff)

        out_terms: list[ObservableTerm] = []
        for (sites, ops), coeff in accum.items():
            if abs(coeff) <= tol:
                continue
            coeff_out: float | complex
            if abs(coeff.imag) <= tol:
                coeff_out = float(coeff.real)
            else:
                coeff_out = coeff
            out_terms.append(ObservableTerm(sites=list(sites), ops=ops, coeff=coeff_out))

        out_terms.sort(key=lambda t: (len(t.sites), tuple(t.sites), t.ops))
        return ObservableSpec(terms=out_terms)

    def commutes_with(self, other: "ObservableSpec", tol: float = 1e-12) -> bool:
        return len(self.commutator(other, tol=tol).terms) == 0


## Interaction term specification
@dataclass(frozen=True)
class InteractionTerm:
    sites: Sequence[int]
    ops: str
    coeff: float = 1.0

    def __post_init__(self) -> None:
        _validate_sites_ops(self.sites, self.ops, allow_empty=False)


## Hamiltonian specification
@dataclass(frozen=True)
class HamiltonianSpec:
    L: int
    terms: Sequence[InteractionTerm]
    boundary: Boundary = "periodic"

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError("L must be >= 2")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        if len(self.terms) == 0:
            raise ValueError("terms must not be empty")

    def to_observable_spec(self) -> "ObservableSpec":
        """Return this Hamiltonian as an ObservableSpec with the same Pauli terms."""
        return ObservableSpec(
            terms=[
                ObservableTerm(
                    sites=list(term.sites),
                    ops=term.ops,
                    coeff=float(term.coeff),
                )
                for term in self.terms
            ]
        )    


## Diagonalized Hamiltonian data class
@dataclass(frozen=True)
class DiagonalizedHamiltonian:
    spec: HamiltonianSpec
    hamiltonian: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        dim = 2**self.spec.L
        if self.hamiltonian.shape != (dim, dim):
            raise ValueError(f"hamiltonian must have shape {(dim, dim)}")
        if self.energies.shape != (dim,):
            raise ValueError(f"energies must have shape {(dim,)}")
        if self.vectors.shape != (dim, dim):
            raise ValueError(f"vectors must have shape {(dim, dim)}")

    # Factory method
    @classmethod
    def from_spec(cls, spec: HamiltonianSpec) -> "DiagonalizedHamiltonian":
        hamiltonian = build_hamiltonian(spec)
        energies, vectors = np.linalg.eigh(hamiltonian)
        return cls(
            spec=spec,
            hamiltonian=hamiltonian,
            energies=energies,
            vectors=vectors,
        )



## Energy window specification
@dataclass(frozen=True)
class EnergyWindow:
    center: float
    width: float

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError("width must be > 0")


## ETH matrix data class
@dataclass
class ETHResult:
    energies: np.ndarray
    diagonal: np.ndarray
    window_indices: np.ndarray | None = None
    offdiag_abs_mean: float | None = None
    offdiag_abs_std: float | None = None
    window_matrix: np.ndarray | None = None


################################################################################################################################
################################################################################################################################
# Internal helper functions for building Hamiltonians and observables, diagonalization, and ETH calculations

def _as_sparse(op: np.ndarray) -> csr_matrix:
    return csr_matrix(op)



#
def _pauli_char_product(lhs: str, rhs: str) -> tuple[complex, str]:
    if lhs == "I":
        return 1.0 + 0.0j, rhs
    if rhs == "I":
        return 1.0 + 0.0j, lhs
    if lhs == rhs:
        return 1.0 + 0.0j, "I"

    table: dict[tuple[str, str], tuple[complex, str]] = {
        ("X", "Y"): (1.0j, "Z"),
        ("Y", "Z"): (1.0j, "X"),
        ("Z", "X"): (1.0j, "Y"),
        ("Y", "X"): (-1.0j, "Z"),
        ("Z", "Y"): (-1.0j, "X"),
        ("X", "Z"): (-1.0j, "Y"),
    }
    return table[(lhs, rhs)]

#
def _multiply_observable_terms(
    lhs: ObservableTerm,
    rhs: ObservableTerm,
) -> tuple[complex, tuple[int, ...], str]:
    coeff = complex(lhs.coeff) * complex(rhs.coeff)

    out_map: dict[int, str] = {}
    for site, op in zip(lhs.sites, lhs.ops):
        if op != "I":
            out_map[int(site)] = op

    for site, op in zip(rhs.sites, rhs.ops):
        site_int = int(site)
        if op == "I":
            continue
        if site_int not in out_map:
            out_map[site_int] = op
            continue

        phase, out_op = _pauli_char_product(out_map[site_int], op)
        coeff *= phase
        if out_op == "I":
            del out_map[site_int]
        else:
            out_map[site_int] = out_op

    if len(out_map) == 0:
        return coeff, tuple(), ""

    ordered_sites = tuple(sorted(out_map.keys()))
    ordered_ops = "".join(out_map[s] for s in ordered_sites)
    return coeff, ordered_sites, ordered_ops


def _site_operator(L: int, site: int, op_char: str) -> csr_matrix:
    if not 0 <= site < L:
        raise ValueError(f"site index out of range: {site}, expected 0 <= site < {L}")

    left = eye(2**site, format="csr", dtype=np.complex128)
    mid = _as_sparse(PAULI[op_char])
    right = eye(2 ** (L - site - 1), format="csr", dtype=np.complex128)
    return kron(kron(left, mid, format="csr"), right, format="csr")


def _term_operator(L: int, sites: Sequence[int], ops: str) -> csr_matrix:
    local = eye(2**L, format="csr", dtype=np.complex128)
    for site, op_char in zip(sites, ops):
        local = local @ _site_operator(L, int(site), op_char)
    return local


# def _bond_count(spec: HamiltonianSpec) -> int:
#     return spec.L if spec.boundary == "periodic" else spec.L - 1



################################################################################################################################
################################################################################################################################
# Weak disorder sampling functions for interaction terms and couplings


def _resolve_rng(
    rng: np.random.Generator | None,
    seed: int | None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("rng and seed cannot both be provided")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def sample_couplings(
    n_terms: int,
    j_min: float,
    j_max: float,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    if j_min > j_max:
        raise ValueError("j_min must be <= j_max")
    generator = _resolve_rng(rng, seed)
    if j_min == j_max:
        return np.full(n_terms, float(j_min), dtype=np.float64)

    return generator.uniform(float(j_min), float(j_max), size=n_terms).astype(np.float64)


def sample_interaction_terms(
    L: int,
    ops: str,
    j_min: float,
    j_max: float,
    boundary: Boundary = "periodic",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> list[InteractionTerm]:
    if L < 2:
        raise ValueError("L must be >= 2")
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'")
    if len(ops) == 0:
        raise ValueError("ops must not be empty")
    allowed = set(PAULI.keys())
    if any(ch not in allowed for ch in ops):
        raise ValueError("ops can only contain I, X, Y, Z")

    term_width = len(ops)
    if term_width > L:
        raise ValueError("len(ops) must be <= L")

    n_terms = L if boundary == "periodic" else L - term_width + 1
    if n_terms < 1:
        raise ValueError("No valid terms for the given L, ops, and boundary")

    coeffs = sample_couplings(n_terms, j_min, j_max, rng=rng, seed=seed)
    terms: list[InteractionTerm] = []
    for start, coeff in enumerate(coeffs):
        if boundary == "periodic":
            sites = [(start + offset) % L for offset in range(term_width)]
        else:
            sites = [start + offset for offset in range(term_width)]
        terms.append(InteractionTerm(sites=sites, ops=ops, coeff=float(coeff)))

    return terms


def sampled_interaction_hamiltonian(
    L: int,
    ops: str,
    j_min: float,
    j_max: float,
    boundary: Boundary = "periodic",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> HamiltonianSpec:
    terms = sample_interaction_terms(
        L=L,
        ops=ops,
        j_min=j_min,
        j_max=j_max,
        boundary=boundary,
        rng=rng,
        seed=seed,
    )
    return HamiltonianSpec(L=L, terms=terms, boundary=boundary)


def all_site_product_observable(
    L: int,
    ops: str,
    coeff: float | complex = 1.0,
) -> ObservableSpec:
    """Return coeff * Prod_i ops_i as an ObservableSpec.

    If ``ops`` is a single Pauli character, the same local operator is used on
    every site. Otherwise ``ops`` must contain one Pauli character per site.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if len(ops) == 1:
        product_ops = ops * L
    elif len(ops) == L:
        product_ops = ops
    else:
        raise ValueError("ops must be a single Pauli character or have length L")

    sites = list(range(L))
    _validate_sites_ops(sites, product_ops, allow_empty=False)
    return ObservableSpec(
        terms=[
            ObservableTerm(
                sites=sites,
                ops=product_ops,
                coeff=coeff,
            )
        ]
    )


def shifted_product_observable(
    L: int,
    ops: str,
    coeff: float | complex = 1.0,
    boundary: Boundary = "periodic",
    tol: float = 1e-12,
) -> ObservableSpec:
    """Return coeff * Prod_i O_i, where O_i is ``ops`` shifted along the chain.

    The product is ordered by increasing start site. For example, ``ops="XY"``
    with periodic boundary conditions gives
    (X_0 Y_1)(X_1 Y_2)...(X_{L-1} Y_0).
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'")
    if len(ops) == 0:
        raise ValueError("ops must not be empty")
    allowed = set(PAULI.keys())
    if any(ch not in allowed for ch in ops):
        raise ValueError("ops can only contain I, X, Y, Z")
    if tol < 0.0:
        raise ValueError("tol must be >= 0")

    term_width = len(ops)
    if term_width > L:
        raise ValueError("len(ops) must be <= L")

    n_factors = L if boundary == "periodic" else L - term_width + 1
    if n_factors < 1:
        raise ValueError("No valid factors for the given L, ops, and boundary")

    product = ObservableSpec(terms=[ObservableTerm(sites=[], ops="", coeff=coeff)])
    for start in range(n_factors):
        if boundary == "periodic":
            sites = [(start + offset) % L for offset in range(term_width)]
        else:
            sites = [start + offset for offset in range(term_width)]
        factor = ObservableSpec(
            terms=[ObservableTerm(sites=sites, ops=ops, coeff=1.0)]
        )
        product = product.multiply(factor, tol=tol)

    return product



################################################################################################################################
################################################################################################################################
# Main functions for building Hamiltonians and observables, diagonalization, and ETH calculations



def build_hamiltonian(spec: HamiltonianSpec) -> np.ndarray:
    L = spec.L
    dim = 2**L
    h = csr_matrix((dim, dim), dtype=np.complex128)

    for term in spec.terms:
        h = h + float(term.coeff) * _term_operator(L, term.sites, term.ops)

    return h.toarray()


def build_observable(L: int, obs: ObservableSpec) -> np.ndarray:
    dim = 2**L
    out = csr_matrix((dim, dim), dtype=np.complex128)

    for term in obs.terms:
        local = _term_operator(L, term.sites, term.ops)
        out = out + complex(term.coeff) * local

    return out.toarray()


def _as_diagonalized(
    hamiltonian: HamiltonianSpec | DiagonalizedHamiltonian,
) -> DiagonalizedHamiltonian:
    if isinstance(hamiltonian, DiagonalizedHamiltonian):
        return hamiltonian
    return DiagonalizedHamiltonian.from_spec(hamiltonian)


def eth_diagonal_elements(vectors: np.ndarray, observable: np.ndarray) -> np.ndarray:
    eig_obs = vectors.conj().T @ observable @ vectors
    return np.diag(eig_obs)


def energy_window_indices(energies: np.ndarray, window: EnergyWindow) -> np.ndarray:
    low = window.center - 0.5 * window.width
    high = window.center + 0.5 * window.width
    return np.where((energies >= low) & (energies <= high))[0]


def eth_elements(
    hamiltonian: HamiltonianSpec | DiagonalizedHamiltonian,
    observable_spec: ObservableSpec,
    window: EnergyWindow | None = None,
    return_full_window_matrix: bool = False,
    max_window_dim: int = 256,
) -> ETHResult:
    if max_window_dim < 1:
        raise ValueError("max_window_dim must be >= 1")

    prepared = _as_diagonalized(hamiltonian)
    energies = prepared.energies
    vectors = prepared.vectors
    observable = build_observable(prepared.spec.L, observable_spec)

    diag = eth_diagonal_elements(vectors, observable)
    result = ETHResult(energies=energies, diagonal=diag)

    if window is None:
        return result

    indices = energy_window_indices(energies, window)
    if indices.size == 0:
        raise ValueError("No states found in the requested energy window")
    if indices.size < 2:
        raise ValueError("At least two states are required for off-diagonal statistics")
    if indices.size > max_window_dim:
        raise ValueError(
            f"Window dimension {indices.size} exceeds max_window_dim={max_window_dim}. "
            "Use a narrower window or increase max_window_dim."
        )

    sub_v = vectors[:, indices]
    window_mat = sub_v.conj().T @ observable @ sub_v
    offdiag_mask = ~np.eye(indices.size, dtype=bool)
    offdiag_abs = np.abs(window_mat[offdiag_mask])

    result.window_indices = indices
    result.offdiag_abs_mean = float(np.mean(offdiag_abs))
    result.offdiag_abs_std = float(np.std(offdiag_abs))

    if return_full_window_matrix:
        result.window_matrix = window_mat

    return result


################################################################################################################################
################################################################################################################################
# Degeneracy check for the energy eigenvalues

def check_energy_degeneracy(
    energy_data: DiagonalizedHamiltonian | ETHResult,
    atol: float = 1e-10,
    rtol: float = 1e-12,
) -> tuple[bool, list[np.ndarray]]:
    """Check whether the energy eigenvalues contain degeneracies.

    ``energy_data`` can be either a DiagonalizedHamiltonian or an ETHResult.
    Returns ``(has_degeneracy, groups)``. Each entry of ``groups`` contains the
    original indices of eigenvalues that are equal within the given tolerance.
    """
    if atol < 0.0:
        raise ValueError("atol must be >= 0")
    if rtol < 0.0:
        raise ValueError("rtol must be >= 0")

    if not isinstance(energy_data, (DiagonalizedHamiltonian, ETHResult)):
        raise TypeError("energy_data must be a DiagonalizedHamiltonian or ETHResult")

    energies = np.asarray(energy_data.energies)
    if energies.size < 2:
        return False, []

    order = np.argsort(energies)
    sorted_energies = energies[order]
    groups: list[np.ndarray] = []
    start = 0
    for index in range(1, sorted_energies.size):
        if np.isclose(
            sorted_energies[index],
            sorted_energies[start],
            atol=atol,
            rtol=rtol,
        ):
            continue
        if index - start > 1:
            groups.append(np.sort(order[start:index]))
        start = index

    if sorted_energies.size - start > 1:
        groups.append(np.sort(order[start:]))

    return len(groups) > 0, groups


################################################################################################################################
################################################################################################################################
# Visualization functions for ETH results


def plot_eth_diagonal_scatter(
    result: ETHResult,
    ax=None,
    figsize=(6.4, 4.8),
    marker_size=5.0,
    alpha=0.5,
    color="red",
    marker="o",
):
    """Scatter plot of ETH diagonal elements against energy eigenvalues.

    Returns ``(fig, ax)``.
    """
    if result.energies.shape != result.diagonal.shape:
        raise ValueError("result.energies and result.diagonal must have the same shape")
    if marker_size <= 0.0:
        raise ValueError("marker_size must be > 0")

    import matplotlib.pyplot as plt

    diagonal = np.real_if_close(result.diagonal, tol=1000)
    if np.iscomplexobj(diagonal):
        raise ValueError("result.diagonal must be real or numerically close to real")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        result.energies,
        diagonal.real,
        s=marker_size,
        c=color,
        marker=marker,
        alpha=alpha,
        edgecolors="none",
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Diagonal elements of observable")

    return fig, ax
