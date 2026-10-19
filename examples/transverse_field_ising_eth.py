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
from standard_models import transverse_field_ising_chain


def main() -> None:
    ham = transverse_field_ising_chain(L=8, j=1.0, h=1.05, g=0.2)
    observable = ObservableSpec([ObservableTerm(sites=[0], ops="Z")])

    diagonalized = DiagonalizedHamiltonian.from_spec(ham)
    result = eth_elements(diagonalized, observable)

    print("lowest energies:", diagonalized.energies[:5])
    print("diagonal elements:", result.diagonal[:5].real)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig, ax = plot_eth_diagonal_scatter(result, color="tab:green")
    ax.set_title("Transverse-Field Ising Chain: ETH Diagonal Elements")
    fig.tight_layout()
    fig.savefig(output_dir / "transverse_field_ising_eth.png", dpi=200)


if __name__ == "__main__":
    main()
