import os
import subprocess

import pandas as pd
import torch
import yaml
from lightning.pytorch import loggers

import metrics


def mlflow_set_environ(mlflow_config_path: dict[str, str]):
    with open(mlflow_config_path["mlflow_auth_path"], "r") as file:
        credentials = yaml.safe_load(file)
    mlflow_tracking_uri = credentials.get("mlflow_tracking_uri")
    mlflow_username = credentials.get("mlflow_username")
    mlflow_password = credentials.get("mlflow_password")
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    return mlflow_tracking_uri


def get_rslab_workstation():
    if os.path.exists("/etc/host_hostname"):
        with open("/etc/host_hostname", "r") as f:
            host_machine = f.read().strip()
    else:
        with open("/etc/hostname", "r") as f:
            host_machine = f.read().strip()
    return host_machine


def log_config_file(logger: loggers.MLFlowLogger, config: dict) -> None:
    logger.experiment.log_dict(
        run_id=logger.run_id,
        dictionary=config,
        artifact_file="config_file.yaml",
    )


def git_cmd(args) -> str | None:
    """Runs a git command and returns the output as a string."""
    try:
        return subprocess.check_output(["git"] + args).decode("utf-8").strip()
    except Exception:
        return None


def log_git_metadata(logger: loggers.MLFlowLogger) -> None:
    """Automatically logs commit, branch, tag, dirty state, and diff."""

    git_metada_dict = dict()

    commit = git_cmd(["rev-parse", "HEAD"])
    if commit:
        git_metada_dict["commit"] = commit

    branch = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        git_metada_dict["branch"] = branch

    tag = git_cmd(["describe", "--tags", "--abbrev=0"])
    if tag and "fatal" not in tag:
        git_metada_dict["tag"] = tag

    dirty = subprocess.call(["git", "diff-index", "--quiet", "HEAD", "--"])
    git_metada_dict["dirty"] = bool(dirty)

    logger.log_hyperparams({"git": git_metada_dict})


def log_cm_and_classification_report(
    logger: loggers.MLFlowLogger,
    cm: torch.Tensor,
    taxonomy: str,
    level: str,
    class_names: list[str],
    pkind: str | None = None,
):
    """Logs the classification report and confusion matrix artifact"""
    if pkind is None:
        pkind = ""
    else:
        pkind = f"-{pkind}"
    cm_df = pd.DataFrame(
        cm.numpy(),
        columns=[f"Pred_{label}" for label in class_names],
    )
    cm_df.insert(0, "", [f"True_{label}" for label in class_names])

    # Log Confusion Matrix
    logger.experiment.log_table(
        run_id=logger.run_id,
        data=cm_df,
        artifact_file=f"confusion_matrix-{taxonomy}-{level}{pkind}.json",
    )

    logger.experiment.log_text(
        run_id=logger.run_id,
        text=metrics.classification_report(cm, class_names),
        artifact_file=f"classification_report-{taxonomy}-{level}{pkind}.txt",
    )
