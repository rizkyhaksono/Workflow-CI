# Workflow-CI

Kriteria 3 — **Workflow CI dengan MLflow Project** (Telco Customer Churn), tingkat **Advanced**.

test

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml      # GitHub Actions: re-training + build & push Docker image
└── MLProject/
    ├── modelling.py              # entry point training (tracking file:./mlruns)
    ├── conda.yaml                # environment definition
    ├── MLProject                 # definisi MLflow Project (entry point main)
    └── telco_preprocessing/      # dataset siap latih
```

## Alur CI (Advanced)

Saat ada `push` ke `main` atau `workflow_dispatch`:

1. `actions/checkout@v3`
2. `setup-python@v5` (Python 3.12.7)
3. **Check Env**
4. **Install dependencies** (mlflow 2.19.0 dll)
5. **Run mlflow project** — `mlflow run . --env-manager local`
6. **Get latest MLflow run_id** — `mlflow.search_runs(...)` → `$GITHUB_ENV`
7. **Install Python dependencies**
8. **Upload to GitHub** — `actions/upload-artifact@v4` (folder `mlruns/`)
9. **Build Docker Model** — `mlflow models build-docker -m runs:/$RUN_ID/model ...`
10. **Log in to Docker Hub** — `docker/login-action@v3`
11. **Tag Docker Image**
12. **Push Docker Image** → Docker Hub

## Secrets yang dibutuhkan

Tambahkan di **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|---|---|
| `DOCKERHUB_USERNAME` | username Docker Hub (rizkyhaksono) |
| `DOCKERHUB_TOKEN` | Personal Access Token Docker Hub |

## Menjalankan lokal (dry-run)

```bash
cd MLProject
mlflow run . --env-manager local
RUN_ID=$(python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); r=mlflow.search_runs(order_by=['start_time DESC'],max_results=1); print(r.loc[0,'run_id'])")
mlflow models build-docker -m "runs:/$RUN_ID/model" -n rizkyhaksono/telco-churn-model:latest --env-manager local
```

## Docker Hub

Image hasil CI: `rizkyhaksono/telco-churn-model:latest`
Tautan: https://hub.docker.com/r/rizkyhaksono/telco-churn-model
