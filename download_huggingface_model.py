import argparse
import joblib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

project_root = str(Path(__file__).parent)


def main(model_id: str):
    # TODO: make sure model is speechbrain
    # Check dirs
    model_dir: str = os.path.join(project_root, model_id)
    if os.path.exists(model_dir):
        user_confirm: str = input(
            f"Model {model_id} already exists in project at {model_dir}!\n"
            "Overwrite? [y]es, any=no",
        )
        if user_confirm in ("y", "yes"):
            pass
        else:
            exit(1)
    else:
        os.makedirs(user_confirm, exist_ok=True)

    # NOTE: Downloading model: https://huggingface.co/docs/hub/models-downloading
    # to add to documentation
