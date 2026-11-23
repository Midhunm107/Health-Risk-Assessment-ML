# Contributing

Three people work in this repository. To avoid stepping on each other, every change goes through a branch and a pull request.

## Who owns what

| Area | Files | Owner |
|------|-------|-------|
| Decision Tree | `notebooks/01_decision_tree.ipynb`, `results/confusion_matrices/dt_*`, `results/visualizations/dt_tree.png` | Adithya Dev |
| Random Forest | `notebooks/02_random_forest.ipynb`, `results/*/rf_*` | Midhun M. Pillai |
| XGBoost | `notebooks/03_xgboost.ipynb`, `results/*/xgb_*` | Midhun M. Pillai |
| WEKA clustering | `notebooks/04_weka_clustering.md` (or notebook), `results/visualizations/weka_*` | Joyal John Mathew |
| Shared code, app, docs | `src/`, `app/`, `docs/`, `README.md` | Anyone, via pull request, reviewed by one other member |

Ownership only says who to ask; anyone can open a pull request against any file.

## Workflow

1. **Add teammates once:** repository → *Settings → Collaborators → Add people* (write access).
2. **Always start from the latest `main`:**

   ```bash
   git checkout main
   git pull
   git checkout -b feature/<short-description>      # e.g. feature/decision-tree-notebook
   ```

3. Make small, focused commits with clear messages (`Add Decision Tree notebook`, `Fix test split in preprocessing`).
4. Push the branch and open a **pull request** to `main`:

   ```bash
   git push -u origin feature/<short-description>
   ```

5. Another member reviews and merges it. Nobody pushes straight to `main`.
6. After merging, everyone runs `git pull` on `main`.

## Rules that keep the results comparable

- **Use `src/preprocessing.py`.** Every model must train on `load_and_split(...)`: the five leakage columns removed,
  stratified 80:20 split, `random_state=42`. Different splits make accuracies incomparable.
- **Save results into `results/`** using the naming pattern `<model>_confusion_matrix.png`, `<model>_feature_importance.png`,
  and add your row to `results/model_comparison.csv`.
- **Do not commit** the dataset, `models/` or `*.joblib` files (they are in `.gitignore`).
- **Keep notebooks clean:** *Runtime → Restart and run all* before saving, so the outputs match the code.
- If you change numbers in `results/`, update `app/results_data.py` and the README table in the same pull request.

## Adding a new algorithm

1. Create `notebooks/NN_<algorithm>.ipynb` (copy the structure of `02_random_forest.ipynb`).
2. Add the model to `src/train_models.py` if the web app should serve it, and to `src/prediction.py`'s `MODEL_FILES`.
3. Add its figures and metrics to `results/` and open a pull request.
