import logging
from openai import OpenAI
import os
import time
import json
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def retrieve_response_batch(batch_id, result_jsonl_file):
    assert not os.path.exists(result_jsonl_file)
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()
    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch job is in status {status}...")

    batch_output_file_id = job.output_file_id

    file_response = client.files.content(batch_output_file_id)
    # print(file_response.text)
    assert not os.path.exists(result_jsonl_file)
    with open(result_jsonl_file, 'wb') as file:
        file.write(file_response.content)

def retrieve_error_batch(batch_id, log_json_file):
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()
    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch job is in status {status}...")

    error_file_id = job.error_file_id
    # Same check for errors
    if error_file_id:
        errors = client.files.content(error_file_id).content.decode("utf-8")
    else:
        logging.info("No errors...")
        return

    with open(log_json_file, 'r') as file:
        old = json.load(file)
    if not "error" in old:
        old["error"] = dict()
    cur_time_str = time.asctime()
    old["error"][cur_time_str] = errors
    with open(log_json_file, 'w') as file:
        json.dump(old, file, indent=4)

if __name__ == "__main__":
    batch_id = "batch_zpMx9vwZ4rMldsR1tdelz7qL"
    log_json_file = "/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context-log.json"
    result_jsonl_file = log_json_file.replace(".json", "")
    result_jsonl_file = result_jsonl_file.replace("-log", '-results')
    result_jsonl_file += ".jsonl"
    retrieve_response_batch(batch_id=batch_id, result_jsonl_file=result_jsonl_file)
    retrieve_error_batch(batch_id=batch_id, log_json_file=log_json_file)