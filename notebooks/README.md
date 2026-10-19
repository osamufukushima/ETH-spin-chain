# Notebooks

## Running the demo

- In Google Colab, open `demo_eth_models.ipynb` using the README's **Open in
  Colab** button. Use a CPU runtime and choose **Runtime > Run all**. The setup
  cell installs the package from the public repository's `main` branch.
- Locally, install the project with `python3 -m pip install -e .` from the
  repository root, then open the notebook with a kernel using that Python
  environment. The Colab installation step is skipped locally.

Each model cell defines its own chain length, and the notebook displays five
plots inline. Most models use `L=6` or `L=8`; the disordered Heisenberg example
uses `L_dis=11` and the observable `Z0 Z1`, matching the README representative
result. Set `L_dis=8` for a faster run.

## Notebook hygiene

Keep public notebooks clean and reproducible:

- clear execution outputs before committing;
- use small system sizes by default;
- save generated figures under `outputs/` rather than embedding large binary data.
