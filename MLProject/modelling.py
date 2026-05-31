"""
modelling.py  (Kriteria 3 - MLflow Project entry point)
=======================================================
Versi modelling untuk dijalankan sebagai MLflow Project di CI (GitHub Actions).
Perbedaan dengan Kriteria 2:
  * Tracking ke local file store (file:./mlruns) -> hermetik, tanpa server.
  * Parameter dapat diatur via argparse (dipakai MLProject entry point).
  * Mencetak RUN_ID agar mudah ditangkap step CI berikutnya.

Dijalankan otomatis oleh: `mlflow run . --env-manager local`
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "telco_train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "telco_test.csv"))
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="telco_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=16)
    args = parser.parse_args()

    # Deteksi apakah script dijalankan di dalam `mlflow run` (MLProject).
    # Jika ya, MLflow sudah membuat run + experiment, jadi JANGAN panggil
    # set_experiment lagi (akan memicu error "active run ID does not match
    # environment run ID"). start_run() tanpa argumen akan melanjutkan run itu.
    in_mlflow_project = "MLFLOW_RUN_ID" in os.environ
    if not in_mlflow_project:
        # Eksekusi langsung (`python modelling.py`): pakai local file store.
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("telco_churn_ci")

    mlflow.sklearn.autolog()

    data_dir = (
        args.data_path
        if os.path.isabs(args.data_path)
        else os.path.join(BASE_DIR, args.data_path)
    )
    X_train, X_test, y_train, y_test = load_dataset(data_dir)

    run_name = None if in_mlflow_project else "rf_ci"
    with mlflow.start_run(run_name=run_name) as run:
        # Tampilkan username Dicoding pada kolom "Created by".
        mlflow.set_tag("mlflow.user", "rizkyhaksono")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", acc)

        run_id = run.info.run_id
        print(f"RUN_ID={run_id}")
        print(f"test_accuracy={acc:.4f}")

        # Tulis run_id ke file agar mudah dibaca step lain bila diperlukan.
        with open(os.path.join(BASE_DIR, "run_id.txt"), "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    main()
