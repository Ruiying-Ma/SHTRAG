# create batch for sht(-r)(-g) and civic, contract, qasper

import json
from openai import OpenAI
import os


def create_batch_sht_civic_civic2_contract_qasper_1000(max_tokens=100, model="gpt-4o-mini"):
    # datasets = ["civic", "civic2", "contract", "qasper"]
    datasets = ["contract"]
    # configs = ["sht", "sht-r", "sht-g", "sht-r-g"]
    # configs = ["sht-abs", "sht-r-abs", "sht-g-abs", "sht-r-g-abs"]
    # configs = ["sht-bm25", "sht-r-bm25", "sht-bm25-g", "sht-r-bm25-g"]
    # configs = ["sht-bm25-abs", "sht-r-bm25-abs", "sht-bm25-g-abs", "sht-r-bm25-g-abs"]
    # configs = ["sht-dpr", "sht-r-dpr", "sht-dpr-g", "sht-r-dpr-g"]
    # configs = ["sht-dpr-abs", "sht-r-dpr-abs", "sht-dpr-g-abs", "sht-r-dpr-g-abs"]
    # configs = ["sht-te3small", "sht-r-te3small", "sht-te3small-g", "sht-r-te3small-g"]
    # configs = ["sht-te3small-abs", "sht-r-te3small-abs", "sht-te3small-g-abs", "sht-r-te3small-g-abs"]
    configs = ["full"]

    json_lines = []
    custom_id_set = set()
    for config in configs:
        for dataset in datasets:
            with open(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/{dataset}-1000.json", 'r') as file:
                queries = json.load(file)
            
            for query in queries:
                query_id = query["id"]
                custom_id = config + ":" + dataset + ":" + str(query_id)
                assert custom_id not in custom_id_set
                custom_id_set.add(custom_id)
                prompt = query["prompt_template"].format(context=query["context"])

                json_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": prompt,
                        }],
                        "max_tokens": max_tokens,
                    }
                })

    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context.jsonl")
    with open("/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context.jsonl", 'w') as file:
        for id, json_line in enumerate(json_lines):
            json.dump(json_line, file)
            if id != len(json_lines) - 1:
                file.write("\n")

def create_batch_raptor_bm25_sbert_dpr_civic_civic2_contract_qasper_1000(max_tokens=100, model="gpt-4o-mini"):
    datasets = ["civic", "civic2", "contract", "qasper"]
    # configs = ["raptor-o", "bm25-o", "sbert-o", "dpr-o"]
    configs = ["te3small", "te3small-o"]

    json_lines = []
    custom_id_set = set()
    for config in configs:
        for dataset in datasets:
            with open(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/{dataset}-1000.json", 'r') as file:
                queries = json.load(file)
            
            for query in queries:
                query_id = query["id"]
                custom_id = config + ":" + dataset + ":" + str(query_id)
                assert custom_id not in custom_id_set
                custom_id_set.add(custom_id)
                prompt = query["prompt_template"].format(context=query["context"])

                json_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": prompt,
                        }],
                        "max_tokens": max_tokens,
                    }
                })
    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/batches/te3small-o-civic-civic2-contract-qasper.jsonl")
    with open("/mnt/f/Research-2024-spring/SHTRAG/batches/te3small-o-civic-civic2-contract-qasper.jsonl", 'w') as file:
    # with open("/mnt/f/Research-2024-spring/SHTRAG/batches/dpr-civic-civic2-contract-qasper.jsonl", 'w') as file:
        for id, json_line in enumerate(json_lines):
            json.dump(json_line, file)
            if id != len(json_lines) - 1:
                file.write("\n")

def upload_batch(jsonl_name):
    MAX_REQUESTS = 50000
    MAX_FILE_SIZE_MB = 100
    FILE_NAME = jsonl_name

    # Get file size in MB
    file_size_mb = os.path.getsize(FILE_NAME) / (1024 * 1024)

    # Check file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise Exception(f"Batch file size is greater than {MAX_FILE_SIZE_MB} MB.")

    # Check number of lines
    with open(FILE_NAME, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)

    if line_count > MAX_REQUESTS:
        raise Exception(f"Batch File has more than {MAX_REQUESTS} requests")

    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(FILE_NAME, "rb"),
        purpose="batch",
    )

    log_jsonl_name = jsonl_name.replace(".jsonl", "")
    log_jsonl_name += "-log"
    log_jsonl_name += ".json"

    with open(log_jsonl_name, 'w') as file:
        json.dump({"upload": batch_input_file.model_dump()}, file, indent=4)


def create_job(batch_input_file_id, descrip, log_json_name):
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()

    with open(log_json_name, 'r') as file:
        old = json.load(file)
    assert "create" not in old

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": descrip,
        }
    )

    
    old["create"] = batch_job.model_dump()
    with open(log_json_name, 'w') as file:
        json.dump(old, file, indent=4)




if __name__ == "__main__":
    # create_batch_sht_civic_civic2_contract_qasper_1000()

    # jsonl_name = "/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context.jsonl"
    # upload_batch(jsonl_name=jsonl_name)

    descrip = '''QA for full context for contract dataset. 1241 tasks.'''
    batch_input_file_id = "file-7q28EWxIKU1kUdQHiaWAdrkz"
    log_json_name = "/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context-log.json"
    create_job(batch_input_file_id=batch_input_file_id, descrip=descrip, log_json_name=log_json_name)