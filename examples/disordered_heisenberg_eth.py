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
from standard_models import disordered_heisenberg_chain


def main() -> None:
    ham = disordered_heisenberg_chain(
        L=11,
        jx=(0.9, 1.1),
        jy=(0.9, 1.1),
        jz=(0.9, 1.1),
        hx=(-0.05, 0.05),
        hy=(-0.05, 0.05),
        hz=(-0.05, 0.05),
        seed=123,
    )
    observable = ObservableSpec([ObservableTerm(sites=[0, 1], ops="ZZ")])

    diagonalized = DiagonalizedHamiltonian.from_spec(ham)
    result = eth_elements(diagonalized, observable)

    print("lowest energies:", diagonalized.energies[:5])
    print("diagonal elements:", result.diagonal[:5].real)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig, ax = plot_eth_diagonal_scatter(result)
    ax.set_title("Disordered Heisenberg Chain: ETH Diagonal Elements")
    fig.tight_layout()
    fig.savefig(output_dir / "disordered_heisenberg_eth.png", dpi=200)


if __name__ == "__main__":
    main()
