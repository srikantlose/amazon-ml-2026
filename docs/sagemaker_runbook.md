# SageMaker runbook

Everything runs on a **SageMaker Notebook Instance** in `us-east-1`, using the `conda_pytorch_p310` kernel.

## 1. Create the instance (console → SageMaker AI → Notebooks → Notebook instances)
- **Instance type:** `ml.g5.xlarge` (A10G 24 GB) or `ml.g4dn.xlarge` (T4 16 GB) for embeddings and fusion. `ml.t3.medium` is only good for editing and GBMs on small data.
- **Volume size: 100 GB.** The default is 5 GB, and 75K + 75K images at 512 px need roughly 8–10 GB before caches. You can't easily shrink it later, but you can raise it while the instance is stopped.
- **Platform identifier:** the latest Amazon Linux 2023 / JupyterLab 4 option.
- **Git repository:** optional. You can link the GitHub repo here, and it gets cloned into `/home/ec2-user/SageMaker/`.
- If the GPU type is greyed out or the instance fails to start, check Service Quotas → SageMaker → "ml.g5.xlarge for notebook instance usage".

## 2. Get the code onto the instance
Only `/home/ec2-user/SageMaker/` survives a stop/start. Keep the repo and data there.
- **Git:** open a terminal and run `cd ~/SageMaker && git clone <repo-url>`.
- **Zip:** locally run `python scripts/make_zip.py`, upload the zip in the Jupyter file browser, then run `cd ~/SageMaker && unzip amazon-ml-2026_code.zip`.

## 3. Environment (repeat after every start, because pip installs into conda envs are not persistent)
Run `notebooks/00_sagemaker_setup.ipynb` with the `conda_pytorch_p310` kernel, or run this in a terminal:
```bash
source activate pytorch_p310
cd ~/SageMaker/amazon-ml-2026
pip install -q -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Kaggle credentials (dry run only): upload `kaggle.json`, then run `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`.
Don't put `kaggle.json` in the repo. It's gitignored and excluded from the zip.

## 4. Long jobs
Browser tabs disconnect, so don't run multi-hour steps in a notebook cell. Use a terminal instead:
```bash
nohup python -m src.embed --config configs/base.yaml > logs/embed.log 2>&1 &
tail -f logs/embed.log
```
Every expensive step caches to `data/`, so a killed job resumes where it stopped (images are skipped if present, embeddings are skipped if cached).

## 5. Day 1 pipeline
```bash
CFG=configs/base.yaml                          # after editing columns/task/metric for 2026
python -m src.download_images --config $CFG --split train   # start first, it's the slowest
python -m src.download_images --config $CFG --split test
python -m src.cv --config $CFG
python -m src.features_text --config $CFG
python -m src.train_gbm --config $CFG           # first submission from here
python -m src.embed --config $CFG
python -m src.train_gbm --config $CFG --features hand svd txt img has_image --name all
python -m src.train_fusion --config $CFG
python -m src.ensemble --config $CFG --runs <run ids> --write
python -m src.validate_submission --config $CFG --sub submissions/<ens id>.csv
```
Dry run on the 2025 data: `bash scripts/dry_run.sh 2>&1 | tee logs/dry_run.log`.

## 6. Cost hygiene
- **Stop** the instance whenever nobody is using it (files are kept). **Delete** removes the volume.
- Never deploy an endpoint. The submission is a CSV produced on the instance.
- Set up a billing budget/alert on every team account.
- Download `submissions/`, `experiments.csv` and `docs/` to a laptop regularly.
