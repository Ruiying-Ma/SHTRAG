import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from utils import write_to_log
import time
from pathlib import Path

def _upload_batch(batch_path):
    log_path_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.upload_batch_job.log", )
    assert "/home/ruiying/SHTRAG/data" in batch_path
    batch_log_path = batch_path.replace("/home/ruiying/SHTRAG/data", log_path_folder).replace(".jsonl", ".file_id.log")
    os.makedirs(os.path.dirname(batch_log_path), exist_ok=True)
    print(batch_log_path)

    MAX_REQUESTS = 100000
    MAX_FILE_SIZE_MB = 200

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
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch",
    )

    
    file_id = batch_input_file.id
    write_to_log(
        log_path=batch_log_path,
        log_entry="Upload\n"+json.dumps(batch_input_file.model_dump())+"\n\n" + f"file_id: {file_id}\n\n"
    )

    print("The id of the uploaded file is", file_id)
    return file_id


def create_batch_job(batch_path, descrip):
    
    batch_input_file_id = _upload_batch(batch_path)

    log_path_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.upload_batch_job.log", )
    assert "/home/ruiying/SHTRAG/data" in batch_path
    batch_log_path = batch_path.replace("/home/ruiying/SHTRAG/data", log_path_folder).replace(".jsonl", ".batch_id.log")
    os.makedirs(os.path.dirname(batch_log_path), exist_ok=True)
    print(batch_log_path)
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        # endpoint="/v1/chat/completions",
        endpoint="/chat/completions",
        completion_window="24h",
        metadata={
            "description": descrip,
        }
    )

    

    batch_id = batch_job.id
    write_to_log(
        log_path=batch_log_path,
        log_entry="Create\n"+json.dumps(batch_job.model_dump())+"\n\n" + f"batch_id: {batch_id}\n\n"
    )
    print('The id of the created is', batch_id)
    return batch_id

if __name__ == "__main__":
    # batch_path = "/home/ruiying/SHTRAG/data/civic/baselines/bm25.gpt-4o-mini.c100.s100/bm25.cosine.raptor0/o0/context0.05/qa_job.jsonl"
    # create_batch_job(
    #     batch_path=batch_path,
    #     descrip=batch_path.replace("/home/ruiying/SHTRAG/data/", "")
    # )


    NODE_EMBEDDING_MODEL_LIST = ["sbert", "dpr", "te3small", "bm25"]
    CONTEXT_LEN_RATIO_LIST = [0.05, 0.1, 0.15, 0.2]
    CONTEXT_CONFIG_LIST = (
        # end-to-end
        [("vanilla", None, nem, None, None, None, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST] + 
        [("raptor", None, nem, None, None, None, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST] + 
        [("sht", None, nem, True, True, True, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST]
        # # ablation on SHT
        # [("sht", sht_type, "sbert", True, True, True, 0.15) for sht_type in SHT_TYPE_LIST] + 
        # # ablation on HI
        # [("sht", None, "sbert", False, True, True, 0.15), ("sht", None, "sbert", True, False, True, 0.15), ("sht", None, "sbert", False, False, True, 0.15)] +
        # # ablation on CI
        # [("sht", None, "sbert", True, True, False, 0.15)]
    )


    batch_num = 0
    for dataset in ["qasper"]:
        for method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks, context_len_ratio in CONTEXT_CONFIG_LIST:
            context_jsonl_path = f"/home/ruiying/SHTRAG/data/{dataset}"
            if method != "sht":
                context_jsonl_path = os.path.join(context_jsonl_path, "baselines")
                context_folder_suffix = f"raptor{int(method=='raptor')}"
            else:
                assert embed_hierarchy != None
                assert isinstance(embed_hierarchy, bool)
                context_folder_suffix = f"h{int(embed_hierarchy == True)}"
            if sht_type != None:
                context_jsonl_path = os.path.join(context_jsonl_path, sht_type)
            query_embedding_model = node_embedding_model
            context_jsonl_path = os.path.join(
                context_jsonl_path,
                f"{node_embedding_model}.gpt-4o-mini.c100.s100",
                f"{query_embedding_model}.cosine.{context_folder_suffix}",
                "o0" if method != "sht" else "l1.h1",
                f"context{context_len_ratio}",
                "context.jsonl"
            )

            assert os.path.exists(context_jsonl_path), context_jsonl_path
            print(context_jsonl_path)

            if context_jsonl_path == "/home/ruiying/SHTRAG/data/civic/baselines/bm25.gpt-4o-mini.c100.s100/bm25.cosine.raptor0/o0/context0.05/context.jsonl":
                continue

            batch_num += 1
            
            create_batch_job(
                batch_path=context_jsonl_path.replace("context.jsonl", "qa_job.jsonl"),
                descrip=context_jsonl_path.replace("/home/ruiying/SHTRAG/data/", "")
            )

    print(batch_num)