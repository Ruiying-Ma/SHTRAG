import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from utils import write_to_log
import time
from pathlib import Path


def retrieve_response(batch_id, result_path):
    assert not os.path.exists(result_path)
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch job is in status {status}...")

    batch_output_file_id = job.output_file_id

    file_response = client.files.content(batch_output_file_id)

    with open(result_path, 'wb') as file:
        file.write(file_response.content)
    print(f'Successfully retrieved {batch_id}')

def retrieve_error(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch job is in status {status}...")

    error_file_id = job.error_file_id
    # Same check for errors
    if error_file_id:
        errors = client.files.content(error_file_id).content.decode("utf-8")
    else:
        print("No errors...")
        return
    
    log_path = os.path.abspath(__file__).replace("3.retrieve_batch_result.py", "error.log")
    write_to_log(
        log_path=log_path,
        log_entry="Retrieve Error\n"+ str(errors) + "\n\n" + f"batch_id: {batch_id}\n\n"
    )