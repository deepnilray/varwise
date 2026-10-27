"""Reproduce and quarantine a fitted VarWise model that failed real validation.

The source data here are the recorded training feature rows only. Held-out
feature rows are read exclusively AFTER .fit() as a reproduction check.
This script cannot grant authorization to scan discovery targets.
"""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from upload import varwise_gaia_hunt as hunt


def main():
    directory = Path(__file__).resolve().parent / "strict_training_real_20261027"
    train_path = directory / "training_features.csv"
    test_path = directory / "sealed_validation_features.csv"
    rep = json.loads((directory / "validation_report.json").read_text())
    if rep.get("validated_for_discovery") is not False:
        raise RuntimeError("Refusing to quarantine a model without a recorded FAILED validation")
    if rep["code_sha256"] != hunt.code_sha256():
        raise RuntimeError("Source code changed since real training; do not reproduce stale model")
    train = pd.read_csv(train_path)
    if len(train) != rep["n_train"]:
        raise RuntimeError("Training record count has changed")
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=.08,
                                         max_iter=200, random_state=0)
    clf.fit(train[hunt.FEATURES], train["label"])
    print(f"Fit reproduced from {len(train)} training curves. Only now checking old holdout.")
    heldout = pd.read_csv(test_path)
    if len(heldout) != rep["n_heldout"]:
        raise RuntimeError("Held-out record count has changed")
    auc = roc_auc_score(heldout["label"], clf.predict_proba(heldout[hunt.FEATURES])[:, 1])
    if abs(auc - rep["auc_heldout"]) > 1e-10:
        raise RuntimeError(f"Reproduction mismatch: {auc} vs {rep['auc_heldout']}")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (train_path, test_path)}
    pack = {"model": clf, "features": hunt.FEATURES,
            "validation_protocol": hunt.VALIDATION_PROTOCOL,
            "validated_for_discovery": False,
            "trained_for_research_only": True,
            "warning": "FAILED HELD-OUT VALIDATION; DO NOT USE FOR DISCOVERY",
            "report": rep, "data_sha256": hashes}
    out = directory / "model_RESEARCH_ONLY_FAILED_VALIDATION.joblib"
    joblib.dump(pack, out)
    if hunt.load_validated_model(out) is not None:
        raise RuntimeError("SAFETY FAILURE: invalid model erroneously authorized!")
    print(f"Preserved quarantined model: {out}")
    print(f"Reproduced held-out AUC {auc:.6f}; discovery loader correctly REJECTS it.")


if __name__ == "__main__":
    main()
