import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download, login
from dotenv import load_dotenv

load_dotenv()

def pull_model():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="The Hugging Face repo ID")
    parser.add_argument("--folder", type=str, required=True, help="Local directory to save the files")
    parser.add_argument("--filename", type=str, default=None, help="Optional: Specific file to download (e.g., 'best.pt')")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF Token")

    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    os.makedirs(args.folder, exist_ok=True)

    if args.filename:
        print(f"⏳ Downloading single file '{args.filename}' from '{args.repo_id}'...")
        file_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.folder,
            repo_type="model"
        )
        print(f"✅ File saved to: {file_path}")
    else:
        print(f"⏳ Downloading entire repo '{args.repo_id}'...")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.folder,
            repo_type="model"
        )
        print(f"✅ Repo downloaded to: {args.folder}")

if __name__ == "__main__":
    pull_model()