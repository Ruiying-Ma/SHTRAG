import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from utils import write_to_log
import time

def create_qa_jobs(
    dataset,
    context_path, # path to the generated context
    max_answer_tokens=100, 
    model="gpt-4o-mini"
):
    assert dataset in context_path
    data_root_dir = os.path.dirname(os.path.abspath(__file__)).replace("batches", "data")
    queries_path = os.path.join(data_root_dir, dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries_info = json.load(file)
    dst = context_path.replace("context.jsonl", "qa_job.jsonl")
    if os.path.exists(dst):
        print(f"{dst} already existsed")
        return
    
    with open(context_path, 'r') as file:
        for l in file:
            context_info = json.loads(l)
            query_id = context_info["id"]
            prompt_template = queries_info[query_id]["prompt_template"]
            assert prompt_template.count("{context}") == 1
            prompt = prompt_template.replace("{context}", context_info["context"])
            job = {
                "custom_id": str(context_info["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "max_tokens": max_answer_tokens,
                }
            }
            with open(dst, 'a') as file:
                contents = json.dumps(job) + "\n"
                file.write(contents)

def create_llmjudge_jobs(
    answer_path, # path to the generated answer
    max_answer_tokens=100, 
    model="gpt-4o-mini"
):
    '''
    Create llm_judge jobs for Qasper
    '''
    dataset = "qasper"
    assert dataset in answer_path
    data_root_dir = os.path.dirname(os.path.abspath(__file__)).replace("batches", "data")
    queries_path = os.path.join(data_root_dir, dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries = json.load(file)
    dst = answer_path.replace("answer.jsonl", "rating_job.jsonl")
    if os.path.exists(dst):
        return

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_judge_prompt.txt"), 'r') as file:
        prompt_template = file.read().strip()

    assert prompt_template.count("{query}") == 1
    assert prompt_template.count("{reference}") == 1
    assert prompt_template.count("{answer}") == 1

    answers = []
    with open(answer_path, 'r') as file:
        for l in file:
            answers.append(json.loads(l))

    assert len(answers) == len(queries)
    
    for answer_info, query_info in zip(answers, queries):
        assert answer_info["id"] == query_info["id"]
        my_answer = answer_info["answer"]
        gold_answers = query_info["answer"]
        for ga_id, gold_answer in enumerate(gold_answers):
            prompt = prompt_template.replace("{query}", query_info['query']).replace("{reference}", gold_answer).replace("{answer}", my_answer)
            job = {
                "custom_id": str(query_info["id"]) + "-" + str(ga_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "max_tokens": max_answer_tokens,
                }
            }
            with open(dst, 'a') as file:
                contents = json.dumps(job) + "\n"
                file.write(contents)

def upload_batch(batch_path):
    MAX_REQUESTS = 50000
    MAX_FILE_SIZE_MB = 100

    # Get file size in MB
    file_size_mb = os.path.getsize(batch_path) / (1024 * 1024)

    # Check file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise Exception(f"Batch file size is greater than {MAX_FILE_SIZE_MB} MB.")

    # Check number of lines
    with open(batch_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)

    if line_count > MAX_REQUESTS:
        raise Exception(f"Batch File has more than {MAX_REQUESTS} requests")
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch",
    )

    log_path = os.path.abspath(__file__).replace("batch.py", "batch.log")
    write_to_log(
        log_path=log_path,
        log_entry="Upload\n"+json.dumps(batch_input_file.model_dump())+"\n"
    )

    file_id = batch_input_file.id
    print("The id of the uploaded file is", file_id)
    return file_id

def create_batch_job(batch_input_file_id, descrip):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": descrip,
        }
    )
    log_path = os.path.abspath(__file__).replace("batch.py", "batch.log")
    write_to_log(
        log_path=log_path,
        log_entry="Create\n"+json.dumps(batch_job.model_dump())+"\n"
    )

    batch_id = batch_job.id
    print('The id of the created is', batch_id)
    return batch_id

def check_full_batch():
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

    print(client.batches.list(limit=10))

def check_batch(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

    job = client.batches.retrieve(batch_id)

    log_path = os.path.abspath(__file__).replace("batch.py", "batch.log")
    write_to_log(
        log_path=log_path,
        log_entry="Check\n"+json.dumps(job.model_dump())+"\n"
    )
    print(f"{batch_id}: {job.status}")
    if job.status == "in_progress":
        print("\t", job.request_counts)
    return job.status

def retrieve_response(batch_id, result_path):
    assert not os.path.exists(result_path)
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

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
    client = OpenAI(api_key=os.getenv("API_KEY"))

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
    
    log_path = os.path.abspath(__file__).replace("batch.py", "batch.log")
    write_to_log(
        log_path=log_path,
        log_entry="Retrieve Error\n"+ str(errors) + "\n"
    )

def cancel_batch(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

    client.batches.cancel(batch_id)

def delete_file(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))

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

if __name__ == "__main__":
    # m_answer_path_id = dict()
    # dataset = "qasper"
    # for query_embedding_model in ["bm25", "sbert", "dpr", "te3small"]:
    #     for is_raptor in [True, False]:
    #         for is_ordered in [True, False]:
    #             # qasper-baselines
    #             answer_path = os.path.join("/home/v-ruiyingma/SHTRAG/data", dataset, "baselines", f"{query_embedding_model}.gpt-4o-mini.c100.s100", f"{query_embedding_model}.cosine.raptor{int(is_raptor)}", f"1000.o{int(is_ordered)}", "answer.jsonl")
    #             assert os.path.exists(answer_path)
    #             create_llmjudge_jobs(
    #                 answer_path=answer_path,
    #             )
    #             file_id = upload_batch(
    #                 batch_path=answer_path.replace("answer.jsonl", "rating_job.jsonl")
    #             )
    #             batch_id = create_batch_job(
    #                 batch_input_file_id=file_id,
    #                 descrip=answer_path
    #             )
    #             m_answer_path_id[answer_path] = {
    #                 "file_id": file_id,
    #                 "batch_id": batch_id
    #             }
    # with open("/home/v-ruiyingma/SHTRAG/batches/temp_contract.json", 'w') as file:
    #     json.dump(m_answer_path_id, file, indent=4)

    # with open("/home/v-ruiyingma/SHTRAG/batches/temp_contract.json", 'r') as file:
    #     jobs = json.load(file)
    # for cpath, ids in jobs.items():
    #     status = check_batch(
    #         batch_id=ids["batch_id"],
    #     )
    #     if status == "completed":
    #         try: 
    #             retrieve_response(
    #                 batch_id=ids["batch_id"],
    #                 result_path=cpath.replace("answer.jsonl", "rating_result.jsonl")
    #             )
    #         except Exception:
    #             continue
    #         delete_file(
    #             batch_id=ids["batch_id"]
    #         )

    
    # create_qa_jobs(
    #     dataset="qasper",
    #     context_path="/home/v-ruiyingma/SHTRAG/data/qasper/intrinsic/sbert.gpt-4o-mini.c100.s100/sbert.cosine.h1/1000.l1.h1/context.jsonl",
    # )

    # file_id = upload_batch(
    #     batch_path="/home/v-ruiyingma/SHTRAG/data/qasper/intrinsic/sbert.gpt-4o-mini.c100.s100/sbert.cosine.h1/1000.l1.h1/qa_job.jsonl"
    # )

    # batch_id = create_batch_job(
    #     batch_input_file_id=file_id,
    #     descrip="/home/v-ruiyingma/SHTRAG/data/qasper/intrinsic/sbert.gpt-4o-mini.c100.s100/sbert.cosine.h1/1000.l1.h1/context.jsonl"
    # )

    # qasper intrinsic
    file_id = "file-7x3VYqKDCQA4nLJALNV9jph5"
    batch_id = "batch_6740705552988190b1d4517f52c5e9a6"
    check_batch(
        batch_id=batch_id
    )

