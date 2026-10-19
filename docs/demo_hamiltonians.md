# Demo Hamiltonians Used In `examples/`

This note explains the Hamiltonians used by the public demo scripts in
`examples/`.

Pauli operators are denoted by $X_i, Y_i, Z_i$ acting on site $i$.

## 1. Transverse-Field Ising Chain

Used in `examples/transverse_field_ising_eth.py`.

$$
H_{\mathrm{TFIM}}
= -j \sum_{i} Z_i Z_{i+1}
  -h \sum_i X_i
  -g \sum_i Z_i
$$

Constructor signature:

```python
transverse_field_ising_chain(
  L: int,
  j: float = 1.0,
  h: float = 1.0,
  g: float = 0.0,
  boundary: Boundary = "periodic",
) -> HamiltonianSpec
```

Parameter specification:

- `L`: chain length.
- `j`: nearest-neighbor Ising coupling for $Z_i Z_{i+1}$.
- `h`: transverse-field strength for $X_i$.
- `g`: longitudinal-field strength for $Z_i$.
- `boundary`: `"periodic"` or `"open"`.

Example:

```python
ham = transverse_field_ising_chain(L=8, j=1.0, h=1.05, g=0.2, boundary="periodic")
```

## 2. XXZ Chain

Used in `examples/xxz_chain_eth.py`.

$$
H_{\mathrm{XXZ}}
= j_{xy} \sum_i \left(X_i X_{i+1} + Y_i Y_{i+1}\right)
  + j_z \sum_i Z_i Z_{i+1}
$$

Constructor signature:

```python
xxz_chain(
  L: int,
  jxy: float = 1.0,
  jz: float = 1.0,
  boundary: Boundary = "periodic",
) -> HamiltonianSpec
```

Parameter specification:

- `L`: chain length.
- `jxy`: planar exchange coupling for $X_i X_{i+1}$ and $Y_i Y_{i+1}$.
- `jz`: Ising-axis coupling for $Z_i Z_{i+1}$.
- `boundary`: `"periodic"` or `"open"`.

Example:

```python
ham = xxz_chain(L=8, jxy=1.0, jz=0.7, boundary="periodic")
```

## 3. Disordered Heisenberg Chain

Used in `examples/disordered_heisenberg_eth.py`.

$$
H_{\mathrm{dis.Heis}}
= \sum_i J_i^{x} X_i X_{i+1}
  + \sum_i J_i^{y} Y_i Y_{i+1}
  + \sum_i J_i^{z} Z_i Z_{i+1}
  + \sum_i (h_i^x X_i + h_i^y Y_i + h_i^z Z_i)
$$

where each channel can be either fixed or weakly disordered. For example,

$$
J_i^x \sim \mathrm{Uniform}[J_x^{\min}, J_x^{\max}], \quad
h_i^x \sim \mathrm{Uniform}[h_x^{\min}, h_x^{\max}],
$$

and similarly for the $y$ and $z$ channels.

Constructor signature:

```python
disordered_heisenberg_chain(
  L: int,
  jx: float | tuple[float, float] = 1.0,
  jy: float | tuple[float, float] = 1.0,
  jz: float | tuple[float, float] = 1.0,
  hx: float | tuple[float, float] = 0.0,
  hy: float | tuple[float, float] = 0.0,
  hz: float | tuple[float, float] | None = None,
  field_strength: float = 0.5,
  boundary: Boundary = "periodic",
  rng: np.random.Generator | None = None,
  seed: int | None = None,
) -> HamiltonianSpec
```

Parameter specification:

- `L`: chain length.
- `jx`, `jy`, `jz`: exchange couplings. Each accepts either:
  - a single float (uniform coupling), or
  - a `(min, max)` tuple (weak disorder range).
- `hx`, `hy`, `hz`: local-field couplings per spin direction. Same input rule
  as above (float or `(min, max)`).
- `field_strength`: backward-compatibility path for Z-field disorder.
  If `hz` is `None`, then $h_i^z \in [-\texttt{field\_strength},\texttt{field\_strength}]$.
- `boundary`: `"periodic"` or `"open"`.
- `rng`: optional NumPy random generator.
- `seed`: optional integer seed.

Important constraint:

- Do not provide both `rng` and `seed` at the same time.

Examples:

```python
ham = disordered_heisenberg_chain(
  L=8,
  jx=(0.9, 1.1),
  jy=(0.9, 1.1),
  jz=(0.9, 1.1),
  hx=(-0.05, 0.05),
  hy=(-0.05, 0.05),
  hz=(-0.05, 0.05),
  seed=123,
)
```

```python
rng = np.random.default_rng(123)
ham = disordered_heisenberg_chain(L=8, jx=(0.9, 1.1), hz=(-0.05, 0.05), rng=rng)
```

## 4. Random Pauli-String Hamiltonian

Used in `examples/random_pauli_hamiltonian.py`.

This demo samples $n_{\mathrm{terms}}$ distinct Pauli strings of fixed width
$w$ on an open chain and builds

$$
H_{\mathrm{rand}}
= \sum_{k=1}^{n_{\mathrm{terms}}} c_k P_k,
$$

where each local term $P_k$ is chosen from translated strings over
$\{X,Y,Z\}^{\otimes w}$, and

$$
c_k \sim \mathrm{Uniform}\left[c_{\min}, c_{\max}\right].
$$

Constructor signature:

```python
random_pauli_hamiltonian(
    L: int,
    term_width: int = 2,
    n_terms: int | None = None,
    coeff_min: float = -1.0,
    coeff_max: float = 1.0,
    seed: int | None = None,
) -> HamiltonianSpec
```

Parameter specification:

- `L`: chain length (`L >= 2`).
- `term_width`: local Pauli-string width (`1 <= term_width <= L`).
- `n_terms`: number of sampled terms.
  - If `None`, it is set to `min(len(all_terms), 2 * L)`.
  - Must satisfy `1 <= n_terms <= len(all_terms)`.
- `coeff_min`, `coeff_max`: uniform coefficient range.
- `seed`: optional integer seed for reproducibility.

Boundary condition behavior:

- This constructor always returns an open-chain Hamiltonian.

Example:

```python
ham = random_pauli_hamiltonian(
    L=6,
    term_width=3,
    n_terms=12,
    coeff_min=-1.0,
    coeff_max=1.0,
    seed=7,
)
```

## 5. Z2 x Z2 Projective XYZ Chain (Eq. 5.14)

Based on Eq. (5.14) of arXiv:2509.01931v2 (the $N=2$ spin-chain example).

Used in `examples/z2z2_projective_xyz_eth.py`.

$$
H_{N=2}
= \sum_j \left(J_j^x X_j X_{j+1} + J_j^y Y_j Y_{j+1} + J_j^z Z_j Z_{j+1}\right)
+ \alpha \sum_j X_j Y_{j+1} Z_{j+2}
$$

with periodic boundary conditions in the paper setup.

Constructor signature:

```python
z2z2_projective_xyz_chain(
  L: int,
  jx: float | tuple[float, float] = 1.0,
  jy: float | tuple[float, float] = 1.0,
  jz: float | tuple[float, float] = 1.0,
  alpha: float = 0.0,
  boundary: Boundary = "periodic",
  rng: np.random.Generator | None = None,
  seed: int | None = None,
) -> HamiltonianSpec
```

Parameter specification:

- `L`: chain length.
- `jx`, `jy`, `jz`: nearest-neighbor exchange couplings for `XX`, `YY`, `ZZ`.
  Each accepts either:
  - a single float (uniform coupling), or
  - a `(min, max)` tuple (weak disorder range).
- `alpha`: coefficient of the three-body `XYZ` term.
- `boundary`: `"periodic"` or `"open"`.
- `rng`: optional NumPy random generator.
- `seed`: optional integer seed.

Important constraints:

- Do not provide both `rng` and `seed` at the same time.

Remarks:

- The extra `XYZ` term is included to break an additional discrete symmetry
  (as discussed around Eq. (5.14) in the paper).
- At `alpha = 0`, this reduces to an XYZ Heisenberg-type chain with weakly
  random channel couplings.

Example:

```python
ham = z2z2_projective_xyz_chain(
  L=11,
  jx=(0.9, 1.0),
  jy=(0.7, 0.8),
  jz=(0.6, 0.7),
  alpha=0.9,
  boundary="periodic",
  seed=123,
)
```

## Mapping To Source

- Definitions live in `standard_models.py`.
- Demos using those definitions live in `examples/`.

## Are These All Parameters?

For the five public demo constructors in `standard_models.py`, yes.
The lists above are complete and include defaults, constraints, and valid
specification patterns.
