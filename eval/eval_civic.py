from utils import *
from typing import List
import json
import re
import os

def civic_q1_eval_answer(
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
    '''
    Evaluate the accuracy of Civic q1.
    '''
    def normalize_answer(s: str):
        return global_normalize_answer(s)

    def is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        assert isinstance(my_answer, str)
        assert isinstance(gold_answer, str)
        delimiters = ["-", "/"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        split_gold_answer = re.split(regex_pattern, gold_answer)
        return all([normalize_answer(ga.strip()) in normalize_answer(my_answer) for ga in split_gold_answer])
    
    
    my_answers = get_answers(
        "civic",
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
    )

    gold_answers = get_gold_answers("civic")

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 380 + 38

    n_correct = 0
    for my_answer, gold_answer in zip(my_answers[:380], gold_answers[:380]):
        assert my_answer["id"] == gold_answer["id"]
        if is_correct_answer(my_answer=my_answer["answer"], gold_answer=gold_answer["answer"]):
            n_correct += 1


    print(f"Civic q1: {n_correct} correct answers among 380 queries\n\taccuracy={round(n_correct * 100 / 380, 3)}")

    return round(n_correct * 100 / 380, 3)

def civic_q2_eval_answer(
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
    '''
    Evaluate the recall, precision and f1 of Civic q2.

    Corner cases:
    https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    '''

    def normalize_project(s: str):
        pattern = r"\(.*?\)"
        new_s = re.sub(pattern, '', s)
        return white_space_fix(remove_punc(lower(normalize_string(new_s))))
    
    def my_projects(my_answer: str, projects: list):
        raw_my_projects = my_answer.replace("[", "").replace("]", "").split(",")
        candid_my_projects = [normalize_project(p.strip()) for p in raw_my_projects]
        my_projects = set(candid_my_projects).intersection(set(projects))
        return my_projects
    
    def gold_projects(gold_answer: list):
        return set([normalize_project(p.strip()) for p in gold_answer])
    
    def eval(my_answer: set, gold_answer: set):
        assert isinstance(my_answer, set)
        assert isinstance(gold_answer, set)
        assert all(isinstance(ma, str) for ma in my_answer)
        assert all(isinstance(ga, str) for ga in gold_answer)
        intersection = gold_answer.intersection(my_answer)
        if len(intersection) == 0:
            if len(gold_answer) == 0 and len(my_answer) == 0:
                recall = 1.0
                precision = 1.0
                f1 = 1.0
            else:
                recall = 0.0
                precision = 0.0
                f1 = 0.0
        else:
            recall = len(intersection) / len(gold_answer)
            precision = len(intersection) / len(my_answer)
            f1 = 2 * recall * precision / (recall + precision)
        return recall, precision, f1

    
    my_answers = get_answers(
        "civic",
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
    )

    gold_answers = get_gold_answers("civic")

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 380 + 38

    projects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), "civic", "projects.json")
    with open(projects_path, 'r') as file:
        m_file_projects = json.load(file)

    tot_recall = 0.0
    tot_precision = 0.0
    tot_f1 = 0.0
    for my_answer, gold_answer in zip(my_answers[380:], gold_answers[380:]):
        assert my_answer["id"] == int(gold_answer["id"])

        file_name = gold_answer["file_name"]
        projects_list = list([normalize_project(p) for p in m_file_projects[file_name].keys()])

        recall, precision, f1 = eval(my_answer=my_projects(my_answer["answer"], projects_list), gold_answer=gold_projects(gold_answer["answer"]))
        tot_recall += recall
        tot_precision += precision
        tot_f1 += f1

    print(f"Civic q2: among 38 queries\n\tavg_recall={round(tot_recall * 100/ 38, 3)}\n\tavg_precision={round(tot_precision * 100/ 38, 3)}\n\tavg_f1={round(tot_f1 * 100/ 38, 3)}")

    return round(tot_recall * 100/ 38, 3), round(tot_precision * 100 / 38, 3), round(tot_f1 * 100/ 38, 3)


if __name__ == "__main__":
    civic_q2_eval_answer(
        chunk_size=100,
        summary_len=100,
        node_embedding_model="sbert",
        query_embedding_model="sbert",
        summarization_model="gpt-4o-mini",
        embed_hierarchy=True,
        distance_metric="cosine",
        context_hierarchy=True,
        context_raw=True,
        context_len=1000,
        is_intrinsic=False,
        is_baseline=False,
        is_raptor=False,
        is_ordered=False,
        is_grobid=True
    )