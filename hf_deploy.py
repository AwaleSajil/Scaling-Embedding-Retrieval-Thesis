import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv("/nas/rhome/sawale/thesis/.env")
token = os.environ["HUGGINGFACE_TOKEN"]
repo_id = "SajilAwale/Embedding-Compression-Eval"
folder = "/nas/rhome/sawale/thesis/eval_v2/outputs/results/nanobeir"

api = HfApi(token=token)

create_repo(repo_id, repo_type="space", space_sdk="static",
            private=False, exist_ok=True, token=token)
print(f"[deploy] repo ready: {repo_id}")

api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=folder,
    allow_patterns=[
        "index.html",
        "README.md",
        "plotly.min.js",
        "significance_shards/*",
        "per_query_shards/*",
    ],
    commit_message="Deploy eval explorer dashboard",
)
print(f"[deploy] DONE → https://huggingface.co/spaces/{repo_id}")
