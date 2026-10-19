# Notebooks

## Running the demo

- In Google Colab, open `demo_eth_models.ipynb` using the README's **Open in
  Colab** button. Use a CPU runtime and choose **Runtime > Run all**. The setup
  cell installs the package from the public repository's `main` branch.
- Locally, install the project with `python3 -m pip install -e .` from the
  repository root, then open the notebook with a kernel using that Python
  environment. The Colab installation step is skipped locally.

The demo uses small systems (`L=6` or `L=8`) and displays five plots inline.

## Notebook hygiene

Keep public notebooks clean and reproducible:

- clear execution outputs before committing;
- use small system sizes by default;
- save generated figures under `outputs/` rather than embedding large binary data.
