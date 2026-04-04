# Deploying DDoS Detection on AWS

## Local macOS note

If `import xgboost` fails with `libomp.dylib`, install OpenMP: `brew install libomp`. The Docker image on Linux includes `libgomp1` so deployment does not need this step.

You need a trained `artifacts/model_bundle.joblib` inside the image (or on a volume). Train locally or in CI before building the image.

## 1. Train and bundle the model

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py --quick
```

The repo default for `--data` is `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (CICIDS2017). Override with `--data /other/file.csv` when needed. The Docker build excludes that CSV via `.dockerignore` to keep images small; copy the CSV into the container or train before building the image.

## 2. Build and push the Docker image

Replace `ACCOUNT`, `REGION`, and `ddos-app` with your values.

```bash
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker build -t ddos-detection .
docker tag ddos-detection:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/ddos-detection:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/ddos-detection:latest
```

## 3. Run on AWS (pick one)

### AWS App Runner

- Create a service from **Container registry** (ECR).
- Set **port** to `8080`.
- Optionally set environment variable `MODEL_PATH=/app/artifacts/model_bundle.joblib` (this is already the default in the Dockerfile).

### Amazon ECS (Fargate)

- Create a task definition using the ECR image.
- Map container port **8080** to the load balancer (e.g. 80).
- CPU/memory: at least **1 vCPU / 2 GB** for XGBoost inference on batch uploads.

### Elastic Beanstalk (Docker)

- Choose **Docker** platform and deploy the same Dockerfile.
- Configure the proxy to forward to port **8080**.

## 4. Optional: train on AWS

For full GridSearch training, run `train_model.py` on a larger EC2 instance or SageMaker, upload `model_bundle.joblib` to S3, then bake it into the image or download at container startup via `entrypoint.sh` (not included; add if you need dynamic model fetch).
