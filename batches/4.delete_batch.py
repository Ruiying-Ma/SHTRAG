import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from utils import write_to_log
import time
from pathlib import Path


def cancel_batch(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    client.batches.cancel(batch_id)

def delete_file(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    batch_input_file_id = job.input_file_id
    output_file_id = job.output_file_id
    error_file_id = job.error_file_id
    
    
    if output_file_id:
        client.files.delete(output_file_id)
    if error_file_id:
        client.files.delete(error_file_id)
    else:
        print("No errors...")
    if batch_input_file_id:
        client.files.delete(batch_input_file_id)
    print(f"Successfully deleted {batch_id}")
