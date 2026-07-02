from __future__ import annotations

# Predictor shared by Streamlit single and batch workflows.
import io
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from sklearn.metrics import pairwise_distances


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "Bagging_final_model.pkl"
TRAINING_PATH = APP_DIR / "data" / "training_fs.csv"

MAX_BATCH_COMPOUNDS = 100
MAX_BATCH_BYTES = 5_000_000

MODEL_METRICS = {
    "training_samples": 474,
    "test_samples": 158,
    "descriptor_count": 24,
    "n_estimators": 20,
    "train_accuracy": 0.9430379747,
    "train_precision": 0.9487179487,
    "train_recall": 0.9367088608,
    "train_mcc": 0.8861469462,
    "test_accuracy": 0.8164556962,
    "test_precision": 0.8205128205,
    "test_recall": 0.8101265823,
    "test_mcc": 0.6329621044,
    "activity_cutoff_pic50": 4.97,
}


class Predictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not TRAINING_PATH.exists():
            raise FileNotFoundError(f"Training descriptors not found: {TRAINING_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.features = [str(name) for name in self.model.feature_names_in_]
        self.calculator = Calculator(descriptors, ignore_3D=True)

        training = pd.read_csv(TRAINING_PATH)
        self.training_x = training[self.features].astype(float)
        self.means = self.training_x.mean(axis=0)
        self.stds = self.training_x.std(axis=0, ddof=0).replace(0, 1.0)
        self.mins = self.training_x.min(axis=0)
        self.maxs = self.training_x.max(axis=0)

        scaled = (self.training_x - self.means) / self.stds
        distances = pairwise_distances(scaled, metric="euclidean")
        np.fill_diagonal(distances, np.inf)
        self.nn_threshold = float(np.quantile(distances.min(axis=1), 0.95))

    def _descriptor_frame(self, mol: Chem.Mol) -> pd.DataFrame:
        result = {str(key): value for key, value in self.calculator(mol).items()}
        values: dict[str, float] = {}
        failures: list[str] = []
        for name in self.features:
            value = result.get(name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                failures.append(name)
                continue
            if not math.isfinite(number):
                failures.append(name)
                continue
            values[name] = number
        if failures:
            raise ValueError(
                "Could not calculate required descriptors: " + ", ".join(failures)
            )
        return pd.DataFrame([values], columns=self.features)

    def _applicability(self, frame: pd.DataFrame) -> dict:
        row = frame.iloc[0]
        z_scores = ((row - self.means) / self.stds).abs()
        range_fraction = float(
            ((row >= self.mins) & (row <= self.maxs)).sum() / len(self.features)
        )
        scaled_query = ((frame - self.means) / self.stds).to_numpy()
        scaled_training = ((self.training_x - self.means) / self.stds).to_numpy()
        nearest = float(pairwise_distances(scaled_query, scaled_training).min())
        distance_ratio = nearest / self.nn_threshold if self.nn_threshold else math.inf

        if range_fraction >= 0.90 and distance_ratio <= 1.0:
            label, level = "Inside domain", "inside"
        elif range_fraction >= 0.75 and distance_ratio <= 1.5:
            label, level = "Near domain edge", "edge"
        else:
            label, level = "Outside domain", "outside"

        top_deviations = [
            {
                "descriptor": str(name),
                "value": round(float(row[name]), 5),
                "|z|": round(float(z_scores[name]), 2),
            }
            for name in z_scores.sort_values(ascending=False).index[:5]
        ]
        return {
            "label": label,
            "level": level,
            "range_fraction": round(range_fraction, 3),
            "nearest_distance": round(nearest, 3),
            "distance_threshold": round(self.nn_threshold, 3),
            "top_deviations": top_deviations,
        }

    @staticmethod
    def _depiction(mol: Chem.Mol) -> bytes:
        image = Draw.MolToImage(mol, size=(720, 480), kekulize=True)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def predict(self, smiles: str, include_depiction: bool = True) -> dict:
        smiles = smiles.strip()
        if not smiles:
            raise ValueError("Enter or draw a structure first.")
        if len(smiles) > 2000:
            raise ValueError("SMILES is too long for this single-compound predictor.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit could not parse this SMILES. Check syntax and valence.")
        if mol.GetNumAtoms() < 2:
            raise ValueError("The structure must contain at least two atoms.")

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        frame = self._descriptor_frame(mol)
        predicted_class = int(self.model.predict(frame)[0])
        class_index = list(self.model.classes_).index(1)
        positive_vote = float(self.model.predict_proba(frame)[0][class_index])

        result = {
            "canonical_smiles": canonical,
            "predicted_class": predicted_class,
            "prediction_label": (
                "Higher-activity class" if predicted_class == 1 else "Lower-activity class"
            ),
            "positive_vote": round(positive_vote, 4),
            "negative_vote": round(1.0 - positive_vote, 4),
            "properties": {
                "MW": round(float(Descriptors.MolWt(mol)), 2),
                "cLogP": round(float(Crippen.MolLogP(mol)), 2),
                "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 2),
                "HBD": int(Lipinski.NumHDonors(mol)),
                "HBA": int(Lipinski.NumHAcceptors(mol)),
                "RotB": int(Lipinski.NumRotatableBonds(mol)),
            },
            "applicability": self._applicability(frame),
        }
        if include_depiction:
            result["depiction_png"] = self._depiction(mol)
        return result

    @staticmethod
    def parse_compound_file(filename: str, content: str) -> list[dict]:
        suffix = Path(filename).suffix.lower()
        compounds: list[dict] = []

        if suffix == ".sdf":
            records = [record for record in content.split("$$$$") if record.strip()]
            for index, record in enumerate(records, start=1):
                mol = Chem.MolFromMolBlock(
                    record.lstrip("\r\n"),
                    sanitize=True,
                    removeHs=False,
                    strictParsing=False,
                )
                if mol is None:
                    compounds.append(
                        {
                            "name": f"Structure {index}",
                            "smiles": "",
                            "parse_error": "Invalid SDF record.",
                        }
                    )
                    continue
                name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                compounds.append(
                    {
                        "name": name or f"Structure {index}",
                        "smiles": Chem.MolToSmiles(
                            mol, canonical=True, isomericSmiles=True
                        ),
                    }
                )
        elif suffix == ".csv":
            frame = pd.read_csv(io.StringIO(content))
            smiles_columns = [
                column for column in frame.columns if "smiles" in str(column).lower()
            ]
            if not smiles_columns:
                raise ValueError("CSV must contain a column whose name includes 'SMILES'.")
            smiles_column = smiles_columns[0]
            name_columns = [
                column
                for column in frame.columns
                if str(column).lower()
                in {"name", "id", "compound", "compound_id", "molecule_id"}
            ]
            name_column = name_columns[0] if name_columns else None
            for index, row in frame.iterrows():
                name = (
                    str(row[name_column]).strip()
                    if name_column and pd.notna(row[name_column])
                    else ""
                )
                smiles = (
                    str(row[smiles_column]).strip()
                    if pd.notna(row[smiles_column])
                    else ""
                )
                compounds.append(
                    {"name": name or f"Compound {index + 1}", "smiles": smiles}
                )
        else:
            for line_number, raw_line in enumerate(content.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if parts[0].lower() in {"smiles", "canonical_smiles"}:
                    continue
                compounds.append(
                    {
                        "name": (
                            parts[1].strip()
                            if len(parts) > 1
                            else f"Compound {line_number}"
                        ),
                        "smiles": parts[0].strip(),
                    }
                )

        if not compounds:
            raise ValueError("No molecular structures were found in the uploaded file.")
        if len(compounds) > MAX_BATCH_COMPOUNDS:
            raise ValueError(
                f"The file contains {len(compounds)} structures; "
                f"the limit is {MAX_BATCH_COMPOUNDS} per batch."
            )
        return compounds

    def predict_batch(self, filename: str, content: str) -> dict:
        compounds = self.parse_compound_file(filename, content)
        rows: list[dict] = []

        for index, compound in enumerate(compounds, start=1):
            base = {
                "index": index,
                "name": compound.get("name") or f"Compound {index}",
                "input_smiles": compound.get("smiles", ""),
            }
            if compound.get("parse_error"):
                rows.append(
                    {**base, "status": "error", "error": compound["parse_error"]}
                )
                continue
            try:
                prediction = self.predict(base["input_smiles"], include_depiction=False)
                rows.append(
                    {
                        **base,
                        "canonical_smiles": prediction["canonical_smiles"],
                        "predicted_class": prediction["predicted_class"],
                        "prediction_label": prediction["prediction_label"],
                        "higher_activity_vote": prediction["positive_vote"],
                        "domain": prediction["applicability"]["label"],
                        "domain_level": prediction["applicability"]["level"],
                        "range_fraction": prediction["applicability"]["range_fraction"],
                        **prediction["properties"],
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append({**base, "status": "error", "error": str(exc)})

        successful = [row for row in rows if row["status"] == "ok"]
        return {
            "filename": Path(filename).name,
            "rows": rows,
            "summary": {
                "total": len(rows),
                "successful": len(successful),
                "failed": len(rows) - len(successful),
                "higher_activity": sum(
                    row["predicted_class"] == 1 for row in successful
                ),
                "inside_domain": sum(
                    row["domain_level"] == "inside" for row in successful
                ),
            },
        }
