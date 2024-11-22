import json
import re
from typing import List
import unicodedata
import string
import os

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
        chunk_size,
        summary_len,
        node_embedding_model,
        query_embedding_model,
        summarization_model,
        embed_hierarchy,
        distance_metric,
        context_hierarchy,
        context_raw,
        context_len,
        is_intrinsic,
        is_baseline,
        is_raptor,
        is_ordered,
        is_grobid,
):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset)
    if is_intrinsic:
        root_dir = os.path.join(root_dir, "intrinsic")
    if is_baseline:
        root_dir = os.path.join(root_dir, "baselines")
    if is_grobid:
        root_dir = os.path.join(root_dir, "grobid")
    if not is_baseline:
        answer_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}", f"{context_len}.l{int(context_raw)}.h{int(context_hierarchy)}")
    else:
        answer_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", f"{context_len}.o{int(is_ordered)}")
    print(answer_dir)
    assert os.path.exists(answer_dir)

    answer_path = os.path.join(answer_dir, "answer.jsonl")
    if not os.path.exists(answer_path):
        response_path = os.path.join(answer_dir, "qa_result.jsonl")
        assert os.path.exists(response_path)
        with open(response_path, 'r') as file:
            for lid, l in enumerate(file):
                response = json.loads(l)
                answer = {
                    "id": int(lid),
                    "answer": response["response"]["body"]["choices"][0]["message"]["content"].strip()
                }
                with open(answer_path, 'a') as file:
                    file.write(json.dumps(answer) + "\n")

    answers = []
    with open(answer_path, 'r') as file:
        for l in file:
            answers.append(json.loads(l))

    return answers


def get_ratings(
        dataset,
        chunk_size,
        summary_len,
        node_embedding_model,
        query_embedding_model,
        summarization_model,
        embed_hierarchy,
        distance_metric,
        context_hierarchy,
        context_raw,
        context_len,
        is_intrinsic,
        is_baseline,
        is_raptor,
        is_ordered,
        is_grobid
):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset)
    if is_intrinsic:
        root_dir = os.path.join(root_dir, "intrinsic")
    if is_baseline:
        root_dir = os.path.join(root_dir, "baselines")
    if is_grobid:
        root_dir = os.path.join(root_dir, "grobid")
    if not is_baseline:
        rating_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}", f"{context_len}.l{int(context_raw)}.h{int(context_hierarchy)}")
    else:
        rating_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", f"{context_len}.o{int(is_ordered)}")
    print(rating_dir)
    assert os.path.exists(rating_dir)

    rating_path = os.path.join(rating_dir, "rating.jsonl")
    if not os.path.exists(rating_path):
        response_path = os.path.join(rating_dir, "rating_result.jsonl")
        assert os.path.exists(response_path)
        responses = []
        with open(response_path, 'r') as file:
            for l in file:
                responses.append(json.loads(l))
        m_qid_ratings = []
        for response in responses:
            rating = response["response"]["body"]["choices"][0]["message"]["content"].strip()
            rating_id = response["custom_id"]
            assert rating_id.count("-") == 1
            query_id = int(rating_id.split("-")[0])
            answer_id = int(rating_id.split("-")[1])
            if query_id == len(m_qid_ratings):
                m_qid_ratings.append([])
            assert query_id == len(m_qid_ratings) - 1
            assert answer_id == len(m_qid_ratings[query_id])
            m_qid_ratings[query_id].append(rating)
        
        for query_id, ratings in enumerate(m_qid_ratings):
            rating_info = {
                "id": int(query_id),
                "rating": ratings
            }
            with open(rating_path, 'a') as file:
                file.write(json.dumps(rating_info) + "\n")
    
    assert os.path.exists(rating_path)

    ratings = []
    with open(rating_path, 'r') as file:
        for l in file:
            ratings.append(json.loads(l))
    return ratings
        
