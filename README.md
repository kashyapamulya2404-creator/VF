
# HR Attrition Streamlit Dashboard

This package contains a single-file Streamlit app (`app.py`) plus a small utils file and a sample dataset.
Drop these files at the root of your GitHub repo and connect the repo to Streamlit Cloud.

## Files in the zip (root of repo)
- app.py            -> main Streamlit app
- utils.py          -> small helper (optional)
- sample_vf.csv     -> tiny example dataset (optional to use)
- requirements.txt  -> packages (no pinned versions)
- README.md         -> this file

## How to deploy
1. Create a new GitHub repository and upload these files (do not place them in folders).
2. On Streamlit Cloud, create a new app from the repo (main file should be `app.py`).
3. Use the default Python and packages. Streamlit Cloud will install packages listed in `requirements.txt`.

## Notes & Features
- Overview tab with 5 charts (interactive filters via sidebar).
- Model Trainer tab to run Decision Tree, Random Forest, Gradient Boosting with stratified CV=5 and simple hyperparameter grids. Generates confusion matrices, ROC, feature importances and a metrics table.
- Predict tab to upload new CSV and generate predictions; download results as CSV.
- The app uses default (unfixed) package versions in `requirements.txt` to reduce version conflicts on Streamlit Cloud.
