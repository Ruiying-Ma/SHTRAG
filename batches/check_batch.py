from openai import OpenAI
import os
import time
import json

def check_all_batches():
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()

    print(client.batches.list(limit=10))

def check_batch(batch_id, log_json_name):
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()

    job = client.batches.retrieve(batch_id)

    with open(log_json_name, 'r') as file:
        old = json.load(file)
    if not "check" in old:
        old["check"] = dict()
    cur_time_str = time.asctime()
    old["check"][cur_time_str] = job.model_dump()
    with open(log_json_name, 'w') as file:
        json.dump(old, file, indent=4)





if __name__ == "__main__":
    # check_all_batches()
    batch_id = "batch_zpMx9vwZ4rMldsR1tdelz7qL"
    log_json_name = "/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context-log.json"
    check_batch(batch_id=batch_id, log_json_name=log_json_name)
