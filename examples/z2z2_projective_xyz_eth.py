from pathlib import Path
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

from eth_spin_chain import (
    DiagonalizedHamiltonian,
    ObservableSpec,
    ObservableTerm,
    eth_elements,
    plot_eth_diagonal_scatter,
)
from standard_models import z2z2_projective_xyz_chain


def main() -> None:
    # Eq. (5.14)-inspired parameters from arXiv:2509.01931v2.
    ham = z2z2_projective_xyz_chain(
        L=11,
        jx=(0.9, 1.0),
        jy=(0.7, 0.8),
        jz=(0.6, 0.7),
        alpha=0.9,
        boundary="periodic",
        seed=123,
    )
    observable = ObservableSpec([ObservableTerm(sites=[0, 1, 2], ops="XYZ")])

    diagonalized = DiagonalizedHamiltonian.from_spec(ham)
    result = eth_elements(diagonalized, observable)

    print("lowest energies:", diagonalized.energies[:5])
    print("diagonal elements:", result.diagonal[:5].real)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig, ax = plot_eth_diagonal_scatter(result)
    ax.set_title("Z2 x Z2 Projective XYZ Chain: ETH Diagonal Elements")
    fig.tight_layout()
    fig.savefig(output_dir / "z2z2_projective_xyz_eth.png", dpi=200)


if __name__ == "__main__":
    main()
