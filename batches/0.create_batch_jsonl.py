from pathlib import Path
import os
import json
import argparse

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
        # print(f"{dst} already existed")
        # return
        assert False, f"{dst} already existed"
    
    job_num = 0
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
                # "url": "/v1/chat/completions",
                "url": "/chat/completions",
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
            job_num += 1
    
    flag = True

    size_bytes = Path(dst).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 200:
        print(f"⚠️ {dst} has {round(size_mb, 2)} MB")
        flag = False
    else:
        print(f"✅ {dst} has {round(size_mb, 2)} MB")

    if job_num > 100000:
        print(f"⚠️ {dst} has {job_num} requests")
        flag = False
    else:
        print(f"✅ {dst} has {job_num} requests")

    if flag == False:
        os.remove(dst)

    return flag    

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
        assert False, f"{dst} already existed"
        # return

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
    
    job_num = 0
    for answer_info, query_info in zip(answers, queries):
        assert answer_info["id"] == query_info["id"]
        my_answer = answer_info["answer"]
        gold_answers = query_info["answer"]
        for ga_id, gold_answer in enumerate(gold_answers):
            prompt = prompt_template.replace("{query}", query_info['query']).replace("{reference}", gold_answer).replace("{answer}", my_answer)
            job = {
                "custom_id": str(query_info["id"]) + "-" + str(ga_id),
                "method": "POST",
                # "url": "/v1/chat/completions",
                "url": "/chat/completions",
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
            job_num += 1

    flag = True

    size_bytes = Path(dst).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 200:
        print(f"⚠️ {dst} has {round(size_mb, 2)} MB")
        flag = False
    else:
        print(f"✅ {dst} has {round(size_mb, 2)} MB")

    if job_num > 100000:
        print(f"⚠️ {dst} has {job_num} requests")
        flag = False
    else:
        print(f"✅ {dst} has {job_num} requests")

    if flag == False:
        os.remove(dst)

    return flag    


if __name__ == "__main__":
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
            
            create_qa_jobs(
                dataset=dataset,
                context_path=context_jsonl_path,
                max_answer_tokens=100,
                model="gpt-4o-mini"
            )

    print(batch_num)