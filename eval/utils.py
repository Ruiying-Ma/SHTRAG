import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
from typing import List
import unicodedata
import string
import config
import logging
import logging_config
import argparse

SAFETY_CHECK = False

def normalize_string(text: str):
    return unicodedata.normalize('NFKC', text)

def remove_articles(text: str):
    return re.sub(r"\b(a|an|the)\b", " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def global_normalize_answer(s: str):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """
    return white_space_fix(remove_articles(remove_punc(lower(normalize_string(s)))))

def get_gold_answers(dataset):
    query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset, "queries.json")
    with open(query_path, 'r') as file:
        queries = json.load(file)
    gold_answers = []
    for q in queries:
        gold_answers.append({
            'id': q["id"],
            "file_name": q["file_name"],
            "answer": q["answer"]
        })
    return gold_answers

def get_answers(
        dataset,
        context_config,
        # chunk_size,
        # summary_len,
        # node_embedding_model,
        # query_embedding_model,
        # summarization_model,
        # embed_hierarchy,
        # distance_metric,
        # context_hierarchy,
        # context_raw,
        # context_len,
        # is_intrinsic,
        # is_baseline,
        # is_raptor,
        # is_ordered,
        # is_grobid,
):
    context_jsonl_path = config.get_config_jsonl_path(dataset, context_config)
    assert os.path.exists(context_jsonl_path), context_jsonl_path
    answer_path = context_jsonl_path.replace("context.jsonl", "answer.jsonl")
    assert os.path.exists(os.path.dirname(answer_path)), answer_path

    # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset)
    # if is_intrinsic:
    #     root_dir = os.path.join(root_dir, "intrinsic")
    # if is_baseline:
    #     root_dir = os.path.join(root_dir, "baselines")
    # if is_grobid:
    #     root_dir = os.path.join(root_dir, "grobid")
    # if not is_baseline:
    #     answer_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}", f"{context_len}.l{int(context_raw)}.h{int(context_hierarchy)}")
    # else:
    #     answer_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", f"{context_len}.o{int(is_ordered)}")
    # print(answer_dir)
    # assert os.path.exists(answer_dir)

    # answer_path = os.path.join(answer_dir, "answer.jsonl")

    if not os.path.exists(answer_path):
        logging.info(f"LLM Response has not been processed to answers yet, processing {context_jsonl_path}...")
        response_path = answer_path.replace("answer.jsonl", "qa_result.jsonl")
        assert os.path.exists(response_path), response_path
        answer_list = []
        with open(response_path, 'r') as file:
            for l in file:
                response = json.loads(l)
                answer_content = response["response"]["body"]["choices"][0]["message"]["content"]
                if isinstance(answer_content, str):
                    answer_content = answer_content.strip()
                answer = {
                    "id": int(response["custom_id"]),
                    "answer": answer_content
                }
                answer_list.append(answer)
        
        M_DATASET_NUM_QUERIES = {
            "civic": 418,
            "contract": 1241,
            "qasper": 1451,
            "finance": 0, # TODO: set finance #queries
        }
        
        missed_answer_id = []
        if (len(answer_list) != M_DATASET_NUM_QUERIES[dataset]):
            answer_id_set = set([a["id"] for a in answer_list])
            query_id_set = set(list(range(M_DATASET_NUM_QUERIES[dataset])))
            logging.info(f"{dataset}, {M_DATASET_NUM_QUERIES[dataset]}, {len(answer_list)}")###################
            assert answer_id_set.issubset(query_id_set), answer_id_set - query_id_set
            missed_answer_id = list(query_id_set - answer_id_set)
            print(f"⚠️ {answer_path} has {len(answer_list)} answers, but the dataset {dataset} has {M_DATASET_NUM_QUERIES[dataset]} queries\n\tmissing {missed_answer_id}")
        for aid in missed_answer_id:
            answer_list.append({    
                "id": int(aid),
                "answer": None,
            })
        
        answer_list = sorted(answer_list, key=lambda x: x["id"])
        assert list(range(len(answer_list))) == [a["id"] for a in answer_list]
        
        if SAFETY_CHECK == True:
            print("Safety check!")
            return None
        
        print(f"✅ Writing answers to {answer_path}...")
        for answer in answer_list:
            with open(answer_path, 'a') as file:
                file.write(json.dumps(answer) + "\n")

    else:
        answer_list = []
        with open(answer_path, 'r') as file:
            for l in file:
                answer_list.append(json.loads(l))
    
    assert [a["id"] for a in answer_list] == list(range(len(answer_list)))

    return answer_list


def get_ratings(
        dataset,
        context_config,
        # chunk_size,
        # summary_len,
        # node_embedding_model,
        # query_embedding_model,
        # summarization_model,
        # embed_hierarchy,
        # distance_metric,
        # context_hierarchy,
        # context_raw,
        # context_len,
        # is_intrinsic,
        # is_baseline,
        # is_raptor,
        # is_ordered,
        # is_grobid
):
    
    context_jsonl_path = config.get_config_jsonl_path(dataset, context_config)
    assert os.path.exists(context_jsonl_path), context_jsonl_path
    rating_path = context_jsonl_path.replace("context.jsonl", "rating.jsonl")
    assert os.path.exists(os.path.dirname(rating_path)), rating_path
    # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset)
    # if is_intrinsic:
    #     root_dir = os.path.join(root_dir, "intrinsic")
    # if is_baseline:
    #     root_dir = os.path.join(root_dir, "baselines")
    # if is_grobid:
    #     root_dir = os.path.join(root_dir, "grobid")
    # if not is_baseline:
    #     rating_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}", f"{context_len}.l{int(context_raw)}.h{int(context_hierarchy)}")
    # else:
    #     rating_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", f"{context_len}.o{int(is_ordered)}")
    # print(rating_dir)
    # assert os.path.exists(rating_dir)

    if not os.path.exists(rating_path):
        logging.info(f"LLM Response has not been processed to ratings yet, processing {rating_path}...")
        response_path = rating_path.replace("rating.jsonl", "rating_result.jsonl")
        assert os.path.exists(response_path)
        responses = []
        with open(response_path, 'r') as file:
            for l in file:
                responses.append(json.loads(l))

        m_qid_aid_ratings = dict()
        for response in responses:
            rating_id = response["custom_id"]
            assert rating_id.count("-") == 1
            query_id = int(rating_id.split("-")[0])
            answer_id = int(rating_id.split("-")[1])
            if query_id not in m_qid_aid_ratings:
                m_qid_aid_ratings[query_id] = dict()
            assert answer_id not in m_qid_aid_ratings[query_id]
            rating_content = response["response"]["body"]["choices"][0]["message"]["content"]
            if isinstance(rating_content, str):
                rating_content = rating_content.strip()
            m_qid_aid_ratings[query_id][answer_id] = rating_content
        
        M_DATASET_NUM_QUERIES = {
            "civic": 418,
            "contract": 1241,
            "qasper": 1451,
            "finance": 0, # TODO: set finance #queries
        }

        if len(m_qid_aid_ratings) != M_DATASET_NUM_QUERIES[dataset]:
            query_id_set = set(list(range(M_DATASET_NUM_QUERIES[dataset])))
            existing_query_id_set = set(m_qid_aid_ratings.keys())
            assert existing_query_id_set.issubset(query_id_set), existing_query_id_set - query_id_set
            missed_query_id = list(query_id_set - existing_query_id_set)
            print(f"⚠️ {rating_path} has {len(m_qid_aid_ratings)} queries with ratings, but the dataset {dataset} has {M_DATASET_NUM_QUERIES[dataset]} queries\n\tmissing {missed_query_id}")
        
        if SAFETY_CHECK == True:
            print("Safety check!")
            return None
        
        rating_list = []
        for qid in range(M_DATASET_NUM_QUERIES[dataset]):
            if qid not in m_qid_aid_ratings:
                rating_info = {
                    "id": qid,
                    "rating": []
                }
            else:
                rating_info = {
                    "id": qid,
                    "rating": list(m_qid_aid_ratings[qid].values())
                }
            assert qid == len(rating_list)
            rating_list.append(rating_info)
            with open(rating_path, 'a') as file:
                file.write(json.dumps(rating_info) + "\n")
        
        return rating_list
    else:    
        assert os.path.exists(rating_path)

        rating_list = []
        with open(rating_path, 'r') as file:
            for l in file:
                rating_list.append(json.loads(l))
        return rating_list
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-check", action='store_true', help="Whether to check the context configures in config.py before continue")
    parser.add_argument("--eval_type", type=str, required=True, choices=["answer", "rating"])
    args = parser.parse_args()
    SAFETY_CHECK = bool(args.safety_check)
    check_context_list = input(f"Check context configure first. Collecting results of qasper {args.eval_type} (safety_check={SAFETY_CHECK}). Do you wanna continue?... [y/n]")
    if check_context_list.lower() != 'y':
        print("Exit")
        exit(0)
    else:
        print("Continue...")

    for context_config in config.CONTEXT_CONFIG_LIST:
        print(context_config)
        if args.eval_type == "answer":
            ratings = get_answers(
                dataset="qasper",
                context_config=context_config,
            )
        else:
            assert args.eval_type == "rating"
            ratings = get_ratings(
                dataset="qasper",
                context_config=context_config,
            )
        