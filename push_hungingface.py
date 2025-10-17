from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./models",
    repo_id="Nuvantim/maxim_sentiment_analysis_model",
    repo_type="model",
)
