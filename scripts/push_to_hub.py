import os
import argparse
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

def push_model():
    parser = argparse.ArgumentParser(description="Push a PyTorch .pt model to the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Target repo ID (e.g., 'username/unet-r18')")
    parser.add_argument("--path", type=str, required=True, help="Path to the local file (best.pt) or folder")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF Token")
    
    args = parser.parse_args()

    if not args.token:
        raise ValueError("❌ Error: HF_TOKEN not found in .env or arguments.")

    login(token=args.token)
    api = HfApi(token=args.token)

    print(f"📦 Ensuring repository '{args.repo_id}' exists...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    # Check if user passed a specific file or a whole folder
    if os.path.isfile(args.path):
        filename = os.path.basename(args.path) # Extracts 'best.pt'
        print(f"🚀 Uploading single file '{filename}' to '{args.repo_id}'...")
        api.upload_file(
            path_or_fileobj=args.path,
            path_in_repo=filename, # Name it will have on Hugging Face
            repo_id=args.repo_id,
            repo_type="model"
        )
    elif os.path.isdir(args.path):
        print(f"🚀 Uploading folder '{args.path}' to '{args.repo_id}'...")
        api.upload_folder(
            folder_path=args.path,
            repo_id=args.repo_id,
            repo_type="model"
        )
    else:
        raise FileNotFoundError(f"❌ Error: The path '{args.path}' does not exist.")

    print(f"✅ Successfully uploaded! https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    push_model()