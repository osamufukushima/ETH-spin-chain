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
from standard_models import random_pauli_hamiltonian


def main() -> None:
    ham = random_pauli_hamiltonian(L=6, term_width=3, n_terms=12, seed=7)
    observable = ObservableSpec([ObservableTerm(sites=[0, 1], ops="XX")])

    diagonalized = DiagonalizedHamiltonian.from_spec(ham)
    result = eth_elements(diagonalized, observable)

    print("number of Hamiltonian terms:", len(ham.terms))
    print("lowest energies:", result.energies[:5])
    print("first diagonal elements:", result.diagonal[:5].real)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig, ax = plot_eth_diagonal_scatter(result, color="tab:purple")
    ax.set_title("Random Pauli Hamiltonian: ETH Diagonal Elements")
    fig.tight_layout()
    fig.savefig(output_dir / "random_pauli_hamiltonian.png", dpi=200)


if __name__ == "__main__":
    main()
