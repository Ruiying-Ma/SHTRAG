import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from utils import write_to_log
import time
from pathlib import Path

def check_all_batches(top_k):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    print(client.batches.list(limit=top_k))


def check_batch(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)

    log_path = os.path.abspath(__file__).replace("2.check_batch_status.py", "batch_status.log")
    write_to_log(
        log_path=log_path,
        log_entry="Check\n"+json.dumps(job.model_dump())+"\n\n" + f"batch_id: {batch_id}\n\n"
    )
    print(f"{batch_id}: {job.status}")
    if job.status == "in_progress":
        print("\t", job.request_counts)
    return job.status

if __name__ == "__main__":
    check_all_batches(100)