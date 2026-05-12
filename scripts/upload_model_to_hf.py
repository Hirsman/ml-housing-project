import os

from huggingface_hub import HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO_ID")  # ex: "username/housing-model"

MODEL_PATH = "backend/models/"


def upload_models():

    if not HF_TOKEN or not REPO_ID:
        raise ValueError("HF_TOKEN ou HF_REPO_ID manquant")

    login(token=HF_TOKEN)

    api = HfApi()

    # upload modèle A
    api.upload_file(
        path_or_fileobj=f"{MODEL_PATH}/model_a.pkl",
        path_in_repo="model_a.pkl",
        repo_id=REPO_ID,
        repo_type="model",
    )

    # upload modèle B
    api.upload_file(
        path_or_fileobj=f"{MODEL_PATH}/model_b.pkl",
        path_in_repo="model_b.pkl",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print("Models uploaded successfully 🚀")


if __name__ == "__main__":
    upload_models()