# TP – Séance 4 : From App to Production
## Fraud Detection API – Deploy, Automate, Retrain

**Tools:** GitHub · Docker · Render · Vercel · GitHub Actions · Apache Airflow
**Goal:** take a trained ML model, expose it as a production API, automate deployment, then set up a nightly ML training pipeline.

---

## Context – What is this application?

You are working for a fintech company that processes thousands of card transactions per day. The fraud team has trained a machine learning model (XGBoost) capable of scoring each transaction in real time and flagging it as fraudulent or legitimate.

The model takes 5 features as input:

| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount in € |
| `hour` | Hour of the transaction (0 = midnight, 23 = 11 PM) |
| `merchant_category` | Category of the merchant (0–4 = common, 5–9 = high-risk) |
| `distance_from_home` | Distance between the transaction location and the cardholder's home (km) |
| `num_transactions_last_24h` | Number of transactions on this card in the last 24 hours |

The model was trained on historical data and currently runs as a script on a data scientist's laptop. **Your job is to take it to production.**

Concretely, you will:
1. Expose the model as a REST API that any application can call
2. Build a Docker image so it runs identically everywhere
3. Automate tests and deployment so every code change goes live safely
4. Host the backend on Render and a simple UI on Vercel — both for free
5. Set up an Airflow pipeline that retrains the model every night on fresh data, with automatic drift detection

---

## About the frontend

The frontend is a **simple static web page** — a single HTML file with a bit of JavaScript. There is no framework (no React, no Vue), no build step, no `npm install`. You open the file in a browser and it works.

**What it does:**
- Displays a form where the user enters the 5 transaction features
- Sends a `POST /predict` request to the backend API when the form is submitted
- Shows the result (fraud / legitimate + probability) returned by the API

**How it communicates with the backend:**

```
Browser (index.html)
      │
      │  POST /predict  {"amount": 150, "hour": 14, ...}
      ▼
Backend API (FastAPI on Render)
      │
      │  {"is_fraud": false, "fraud_probability": 0.02, ...}
      ▼
Browser displays the result
```

The API URL is configured in `config.js`:

```js
const CONFIG = {
    API_URL: "https://YOUR-APP.onrender.com",  // ← your Render URL
};
```

This file is the **only thing you need to edit** in the frontend. Everything else is already written.

**Why Vercel?** Vercel hosts static files for free and redeploys automatically every time you push to the `prod` branch — exactly like Render does for the backend. You don't write any Vercel configuration.

> You are not evaluated on the frontend. Its only purpose is to give a visual interface to test your production API.

---

## Architecture

```
Developer
    │
    git push → prod branch
                    │
            GitHub Actions (CI/CD)
                    │
          ┌─────────┴──────────┐
          │  pytest (tests)    │
          │  if OK ↓           │
          │  Render deploy     │
          └─────────┬──────────┘
                    │
         Backend API (Render)          Frontend (Vercel)
         FastAPI + XGBoost        ←──  HTML / JS
         /predict  /health             reads from config.js
                    ▲
                    │  (scheduled – configured in Part 5)
         Airflow Training Pipeline
              extract → validate → check_drift
                                       ↓ (if drift)
                                   preprocess → train → evaluate → save
```

---

## Project structure

```
TP_Seance4/
├── backend/
│   ├── main.py               ← FastAPI app          [PROVIDED]
│   ├── requirements.txt      ← Python dependencies  [PROVIDED]
│   ├── model/
│   │   └── train.py          ← Training script      [PROVIDED]
│   └── tests/
│       └── test_api.py       ← Deployment tests     [PROVIDED]
│
├── frontend/
│   ├── index.html            ← UI                   [PROVIDED]
│   └── config.js             ← API URL config       [edit this]
│
├── airflow/
│   ├── dags/
│   │   └── fraud_retrain_dag.py  ← Airflow DAG      [PROVIDED]
│   ├── Dockerfile                ← Airflow image     [PROVIDED]
│   └── docker-compose.airflow.yml                    [PROVIDED]
│
└── .github/
    └── workflows/
        └── deploy.yml        ← CI/CD skeleton       [write both jobs]
```

**What YOU must write:**
- `backend/Dockerfile`
- `.github/workflows/deploy.yml` — both the `test` job and the `deploy-backend` job
- Render configuration (web dashboard)
- Vercel configuration (web dashboard)
- GitHub secrets (`RENDER_DEPLOY_HOOK`)

---

## Part 1 – Local setup

### 1.1 Clone and install

```bash
git clone <your-repo>
cd TP_Seance4

cd backend
pip install -r requirements.txt
```

### 1.2 Train the model

```bash
# From backend/
python model/train.py
# Expected: Test AUC: 0.97xx
# Creates: model/fraud_model.pkl  +  model/baseline_stats.json
```

### 1.3 Run the API locally

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs → test the `/predict` endpoint with the Swagger UI.

### 1.4 Run the tests

```bash
pytest tests/ -v
```

Expected: all 8 tests pass.

---

## Part 2 – Dockerization

### 2.1 Write `backend/Dockerfile`

Your Dockerfile must:
- Start from `python:3.11-slim`
- Copy `requirements.txt` and install dependencies
- Copy all backend files
- Train the model at build time (`python model/train.py`)
- Expose port 8000
- Launch the API with `uvicorn`



### 2.2 Test your Dockerfile

```bash
docker build -t fraud-api ./backend
docker run -p 8000:8000 -e MODEL_VERSION=v1.0 fraud-api
```

Check: `curl http://localhost:8000/health`

---

## Part 3 – GitHub & CI/CD

### 3.1 Create your GitHub repository

```bash
git init
git remote add origin https://github.com/<you>/fraud-detection.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 3.2 Create the `prod` branch

```bash
git checkout -b prod
git push -u origin prod
```

> All future production deployments happen via **push to `prod`**.
> You develop on `main` or feature branches, then merge to `prod` to deploy.

### 3.3 Write the GitHub Actions workflow

The file `.github/workflows/deploy.yml` contains a skeleton with two jobs and their `TODO` comments. **You must write both jobs yourself.**

**Job `test`** — runs on every push and every PR targeting `prod`:
- Set up Python
- Install the backend dependencies
- Train the model (the tests need it)
- Run the tests against the source code directly (not inside the container)

**Job `deploy-backend`** — runs only on push to `prod`, after `test` passes:
- Call the Render deploy hook to trigger a redeployment
- The URL must be stored as a GitHub secret, not hardcoded

Things to think about:
- How do you tell a job to wait for another job to succeed before starting?
- How do you reference a secret inside a `run:` command?
- The `working-directory` option lets you run a step from a specific folder

You will add the Render secret in Step 4.2, once your Render service is created.

---

## Part 4 – Deploy on Render + Vercel

### 4.1 Backend → Render

1. Go to https://render.com → New → **Web Service**
2. Connect your GitHub repository
3. Configure the service so that Render uses **your Dockerfile** to build and run the container
   - Set the branch to `prod`
   - Set the plan to **Free**
   - Add the environment variable `MODEL_VERSION = v1.0`
4. Deploy and note your URL: `https://YOUR-APP.onrender.com`

> **Hint:** Render supports deploying Docker containers directly.
> Look for the right environment type when creating the service.

**Verify your production API is live:**
```bash
curl https://YOUR-APP.onrender.com/health
```

### 4.2 Add the Render deploy hook (GitHub Actions)

1. In Render: your service → **Settings** → **Deploy Hook** → copy the URL
2. In GitHub: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
   - Name: `RENDER_DEPLOY_HOOK`
   - Value: the URL from Render
3. Push to `prod` → watch the Actions tab → verify Render redeploys

### 4.3 Frontend → Vercel

1. Edit `frontend/config.js`:
   ```js
   const CONFIG = {
       API_URL: "https://YOUR-APP.onrender.com",   // ← your real URL
   };
   ```
2. Go to https://vercel.com → New Project → Import your GitHub repo
3. Configure:
   - **Framework:** Other (static site)
   - **Root directory:** `frontend`
   - **Branch:** `prod`
4. Deploy → visit your Vercel URL

> Vercel auto-redeploys every time you push to `prod`.

### 4.4 Test the full chain

1. Make a small change to `frontend/index.html` (e.g. change the title)
2. Push to `prod`
3. Verify:
   - GitHub Actions runs and passes
   - Render redeploys the backend
   - Vercel redeploys the frontend
   - The change appears on your Vercel URL

---

## Part 5 – Airflow ML Pipeline

The DAG (`airflow/dags/fraud_retrain_dag.py`) is provided. Your task is to:
1. Start Airflow locally
2. Manually trigger and validate the pipeline
3. Configure the automatic schedule and observe it trigger on its own
4. Understand the drift detection logic

### How Airflow connects to your project

Airflow runs inside Docker containers and cannot see your files by default.
The connection is made entirely through **Docker volume mounts** defined in `airflow/docker-compose.airflow.yml`.
Each mount maps a folder on your machine to a path inside the container:

```
Your machine                          Inside the Airflow container
────────────────────────────────────  ──────────────────────────────────
airflow/dags/                    →    /opt/airflow/dags/
  fraud_retrain_dag.py                  DAG read live – edits are instant

data/                            →    /opt/airflow/data/
  transactions.csv                      written by extract_transactions task

backend/model/                   →    /opt/airflow/model/
  fraud_model.pkl                       read/written by train + save tasks
  baseline_stats.json                   read by check_data_drift task
```

> The paths on your machine are resolved **relative to the `airflow/` folder** (where the compose file lives), so `../backend/model` always points to `backend/model/` regardless of where you run the command from.

The DAG reads the container-side paths via environment variables (`DATA_PATH`, `MODEL_PATH`, `BASELINE_PATH`) defined in the compose file.

> **Read `airflow/docker-compose.airflow.yml` now.** Every line is commented — make sure you understand what each volume, environment variable, and service does before moving on.

### 5.1 Start Airflow

```bash
docker-compose -f airflow/docker-compose.airflow.yml up --build
```

The `--build` flag builds the custom Airflow image (with ML dependencies baked in) on the first run. Subsequent runs reuse the cached image and start instantly.

Wait ~2 min, then open http://localhost:8080
Login: `admin` / `admin`

> First run only: `airflow-init` creates the DB and user. It will exit with code 0 – that's normal.

### 5.2 Explore the DAG

In the Airflow UI:
1. Find `fraud_model_nightly_retrain`
2. Click **Graph** → study the task dependencies before running anything
3. Identify the two branches: what conditions lead to each path?

### 5.3 Manual trigger – validate the pipeline works

Before setting up automatic scheduling, you must confirm the pipeline runs correctly end-to-end.

1. Enable the DAG toggle (OFF → ON)
2. Click **Trigger DAG ▶** to start a manual run
3. Watch the tasks execute in the Graph view
4. Check that every task succeeds and identify which branch `check_data_drift` took — and why
5. Trigger the DAG a second time and observe whether the branch changes

**Do not move on to 5.4 until you have at least two successful manual runs.**

### 5.4 Configure the automatic schedule

The DAG currently has `schedule_interval=None` — it only runs when triggered manually.

**Your task:** change the `schedule_interval` in `fraud_retrain_dag.py` so the DAG triggers automatically at **19:00 in your local timezone**.

Things to think about:
- Airflow schedules use UTC by default — what UTC time corresponds to 19:00 in your timezone?
- Cron format: `minute hour * * *`
- After saving the file, Airflow picks up the change automatically (no restart needed)

Once configured, keep the Airflow UI open and **wait for the automatic trigger at 19:00**. You should see a new run appear in the DAG history without having clicked anything.

### 5.5 Understand the drift detection

Open `airflow/dags/fraud_retrain_dag.py` and answer:

1. What does the `check_data_drift` task return when no baseline exists?
2. Which statistical test is used, and what does the p-value threshold mean?
3. What happens if `validate_data` fails (e.g. only 100 rows)?
4. Why does `evaluate_auc` raise a `ValueError` instead of just logging?
5. What does `save_to_registry` currently do instead of a real registry?

### 5.6 Observe the branch (exercise)

Force a drift by editing `extract_transactions` to change the `amount` distribution:

```python
# Original: np.random.exponential(60, n_legit)
# Change to (simulates drift):
"amount": np.random.exponential(300, n_legit),   # ← 5x higher amounts
```

Trigger the DAG manually → the `check_data_drift` task should now take the **retrain** branch.


## Additional Exercises

These exercises extend the core TP. 

---

### Exercise 1 – Production smoke test in CI/CD

Right now, the GitHub Actions pipeline deploys to Render and stops there.
The problem: a successful deploy doesn't guarantee the API is actually responding.

**Your task:** add a step at the end of the `deploy-backend` job that automatically calls the `/health` endpoint of your Render service and fails the pipeline if the API does not respond correctly.

Things to think about:
- Render takes a few seconds to restart the container after a deploy — your step needs to account for that
- The Render URL must not be hardcoded in the workflow file
- The step should fail the whole job if the API returns anything other than a successful response

---

### Exercise 2 – Model versioning

Currently, every deployment serves `MODEL_VERSION=v1.0` forever, even after the Airflow pipeline retrains and promotes a new model.

**Your task:** make the version meaningful and traceable end-to-end:
- Each time the Airflow DAG successfully promotes a new model, the version must be updated automatically (e.g. using the training date or an incremental counter)
- The `/health` endpoint must always return the version of the model currently in memory
- After a successful retraining cycle, the version visible in `/health` must be different from the previous one

Things to think about:
- Where is the version stored so that both the Airflow DAG and the API can access it?
- The running API loads the model once at startup and keeps it in memory — replacing the `.pkl` file on disk has no effect on the live container. The only way to serve the new model is to restart the container.
- On Render, restarting the container means triggering a redeploy. The Render deploy hook (already used in your CI/CD pipeline) can also be called from the Airflow DAG at the end of `save_to_registry`, using a simple HTTP POST — this closes the full loop: retrain → promote → redeploy production API automatically.

**Suggested approach:**

1. Add a `POST /reload-model` endpoint to the FastAPI app that reloads the model from disk without restarting the server — this avoids a full redeploy when the model file is updated via the shared volume.

2. Store the version in a `fraud_model_meta.json` file written by `save_to_registry` alongside the model (e.g. `trained_at` timestamp). The API reads this file when loading the model and returns the value in `/health`.

3. At the end of `save_to_registry`, call `POST /reload-model` on the API URL so the live server picks up the new model immediately. The API URL should be configured via an environment variable (`API_URL`) so it works both locally (`http://host.docker.internal:8000`) and in production (`https://YOUR-APP.onrender.com`).

The full loop locally:
```
Airflow save_to_registry
  → writes fraud_model.pkl + fraud_model_meta.json  (shared volume)
  → POST /reload-model
      → API reloads model from disk
      → GET /health now returns the new version
```

---

### Exercise 3 – Data versioning with DVC

The Airflow pipeline currently generates new data on every run without tracking it.
In a real MLOps setup, every training run must be reproducible: given a model, you must be able to retrieve the exact dataset it was trained on.

**Your task:** integrate [DVC](https://dvc.org) into the project to version the training data alongside the code.

What you need to do:
- Initialize DVC in the repository and configure a local remote (a folder on your machine is enough)
- Track the training data file (`data/transactions.csv`) with DVC so that `git status` no longer shows it as untracked
- Add a step in the Airflow DAG (or as a separate script called by the DAG) that runs `dvc push` after the data is extracted, so each training run's dataset is saved to the remote
- Verify that you can reproduce a past training run by checking out a previous Git commit and running `dvc pull` to restore the corresponding data

Things to think about:
- DVC separates data versioning from code versioning: the `.dvc` file goes in Git, the data goes in the DVC remote
- You do not need a cloud remote for this exercise — a local folder works fine
- How would you extend this to also version the trained model artifact?

---

### Exercise 4 – Faster builds and deployments

Every push to `prod` triggers three slow steps: installing Python dependencies in CI, building the Docker image, and pushing it to Render. This exercise asks you to cut that time by applying three independent optimisations.

---

**Optimisation 1 – Replace `pip` with `uv`**

[`uv`](https://github.com/astral-sh/uv) is a Python package installer written in Rust. It is a drop-in replacement for `pip` and is typically 10–100× faster, with built-in caching and dependency resolution.

Replace `pip` with `uv` in two places:

*In `backend/Dockerfile`:*
- Add `uv` to the image (there is an official way to copy just the binary from the `ghcr.io/astral-sh/uv` image)
- Replace the `pip install` step with `uv pip install`

*In `.github/workflows/deploy.yml`:*
- Use the official `astral-sh/setup-uv` action to install `uv` on the runner
- Replace the `pip install` step with `uv pip install`
- Enable uv's built-in cache so repeated runs skip already-resolved packages

Things to think about:
- `uv pip install --system` is needed when not inside a virtual environment (e.g. in Docker or CI)
- The `setup-uv` action supports a `cache-dependency-glob` parameter — what file should it watch?

---

**Optimisation 2 – Add a `.dockerignore`**

Without it, every `docker build` sends the entire project directory to the Docker daemon as build context — including `model/*.pkl`, `__pycache__`, `.git`, test fixtures, etc. This bloats the context and can invalidate layer caches unnecessarily.

Your task: create `backend/.dockerignore` and list the files and folders that the Docker build does not need.

Things to think about:
- What is already generated at build time inside the container and should not be copied in?
- What development artifacts (cache folders, test outputs) are irrelevant at runtime?

---

**Optimisation 3 – Cache Docker layers in GitHub Actions**

Right now, every CI run rebuilds the Docker image from scratch, even if only `main.py` changed. The `RUN uv pip install` layer — the slowest one — is rebuilt every time because GitHub Actions runners start with an empty Docker cache.

Your task: update the `deploy-backend` job to build the image using `docker/build-push-action` with GitHub Actions cache enabled, so the dependency layer is reused across runs when `requirements.txt` has not changed.

Things to think about:
- `docker/build-push-action` supports `cache-from` and `cache-to` parameters — which cache type works without a registry?
- The `load: true` option is needed if you want to run the image locally after building it in CI

---

**Measure and report the observed speedup:**

Fill in the table with the durations you observe in the GitHub Actions tab:

| | before | after (cold cache) | after (warm cache) |
|---|---|---|---|
| Install dependencies (CI) | | | |
| Docker build – dep layer (CI) | | | (cached) |

> A difference of less than 3× on the install step is a sign that the uv cache is not correctly enabled.
> A difference of less than 2× on the Docker build is a sign that the layer cache is not being hit.

---

## Tips & Troubleshooting

| Problem | Fix |
|---------|-----|
| Render says "Service Unavailable" | Check logs in Render dashboard – model may not be trained |
| `pytest` fails on `import main` | Run from inside `backend/` folder |
| Airflow scheduler doesn't pick up DAG | Check `/opt/airflow/dags/` path in the volume mount |
| Vercel shows old page | Hard-refresh (Ctrl+Shift+R) or check build logs in Vercel |
| `RENDER_DEPLOY_HOOK` secret missing | Add it in GitHub → Settings → Secrets → Actions |
| KS test always shows no drift | Increase the amount shift or lower `DRIFT_P_VALUE` to 0.1 |

---

*MLOps · Séance 4 · ECE · 2026*
