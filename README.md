# SA-inhibitor-Predictor 

Prioritization of candidate for *Staphylococcus aureus* inhibitors.

## Included workflows

- Draw a structure with the embedded Ketcher editor.
- Paste a single SMILES string.
- Upload up to 100 structures in `.smi`, `.smiles`, `.txt`, `.csv`, or `.sdf`.
- Download batch results as CSV.
- Review ensemble support, molecular properties, and applicability domain.
- View separate training and held-out test statistics.

## Local validation

Use Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`.


## Model interpretation

- Algorithm: BaggingClassifier with 20 decision trees.
- Inputs: 24 selected Mordred 2D descriptors.
- Training performance: accuracy 0.9430, precision 0.9487, recall 0.9367,
  MCC 0.8861.
- Held-out test performance: accuracy 0.8165, precision 0.8205, recall 0.8101,
  MCC 0.6330.
- Class 1 is the higher-pIC50 half of the supplied dataset, split near pIC50
  4.97.
- Ensemble support is not a calibrated probability.

## Scope

This application is for research prioritization only. It does not establish
MIC, bactericidal activity, selectivity, toxicity, pharmacokinetics, or clinical
efficacy.
