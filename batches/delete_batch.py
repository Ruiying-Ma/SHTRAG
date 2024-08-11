import os
from openai import OpenAI
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def cancel_batch(batch_id):
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()
    client = OpenAI()

    client.batches.cancel(batch_id)

def delete_file(batch_id):
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    client = OpenAI()
    job = client.batches.retrieve(batch_id)
    batch_input_file_id = job.input_file_id
    output_file_id = job.output_file_id
    error_file_id = job.error_file_id
    
    
    if output_file_id:
        client.files.delete(output_file_id)
    if error_file_id:
        client.files.delete(error_file_id)
    else:
        logging.info("No errors...")
    if batch_input_file_id:
        client.files.delete(batch_input_file_id)

if __name__ == "__main__":
    batch_id = "batch_8v2n5SxGqYfsR8qYTtdJtXq4"
    # cancel_batch(batch_id)
    delete_file(batch_id)
