# UFC Predictor Environment Guide

1. Create the environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it:
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\\Scripts\\Activate.ps1
     ```
3. Install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

After activation, open `ufcstats_scraper.ipynb` in Jupyter Lab/Notebook and run the cells in order.
