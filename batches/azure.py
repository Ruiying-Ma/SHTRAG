import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

def azure(job_path, response_path):
    assert job_path.endswith(".jsonl")
    assert response_path.endswith(".jsonl")
    assert os.path.exists(job_path)
    assert not os.path.exists(response_path)
    jobs = []
    with open(job_path, 'r') as file:
        for l in file:
            jobs.append(json.loads(l))

    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_VERSION"),
        api_key=os.getenv("AZURE_API_KEY")
    )

    for job in jobs:
        assert all([key in job for key in ["body", "custom_id"]])
        assert all([key in job["body"] for key in ["model", "messages", "max_tokens"]])
        response = client.chat.completions.create(
            model=os.getenv("AZURE_MODEL"),
            messages=job["body"]["messages"],
            max_tokens=job["body"]["max_tokens"]
        )
        response_info = {
            "custom_id": job["custom_id"],
            "response": {
                "body": response.to_dict()
            }
        }
        custom_id = job["custom_id"]
        print(f"Responded to {custom_id}")
        with open(response_path, 'a') as file:
            file.write(json.dumps(response_info) + "\n")