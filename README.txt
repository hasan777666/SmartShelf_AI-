# SMARTSHELF AI v3

## Run on Windows (VS Code terminal)
1. Extract this ZIP into a new folder (keep your old project as a backup).
2. Open the extracted `SMARTSHELF_AI_v3` folder in VS Code.
3. In the terminal, activate your existing virtual environment if you copied it, or create one:
   `python -m venv venv`
4. Activate:
   `.\venv\Scripts\Activate.ps1`
5. Install:
   `pip install -r requirements.txt`
6. Run:
   `streamlit run app.py`

## Features
- Sales dashboard
- Baseline demand estimate
- Size and color insights
- Smart replenishment recommendations with CSV export
- Business advisor

The included data is synthetic demo data. Forecast and replenishment outputs are estimates, not trained-ML predictions or purchasing guarantees.
