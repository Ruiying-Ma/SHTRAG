import json
import re
from typing import List
import unicodedata
import string

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


def civic_eval_answer_accuracy(configs):

    def civic_normalize_answer(s: str):
        return global_normalize_answer(s)

    def civic_is_correct_answer(my_answer: str, gold_answer: List[str]) -> bool:
        assert isinstance(my_answer, str)
        assert isinstance(gold_answer, List)
        assert all([isinstance(a, str) for a in gold_answer])
        return all([civic_normalize_answer(a) in civic_normalize_answer(my_answer) for a in gold_answer])

    

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/civic.json", 'r') as file:
        queries = json.load(file)

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/civic.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_correct_answer_num = dict()
    for config in configs:
        correct_answer_num = 0
        my_answers = m_config_to_answers[config]
        assert len(my_answers) == len(queries)
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])
            
            delimiters = ["-", "/"]
            regex_pattern = "|".join(map(re.escape, delimiters))
            raw_gold_ans = re.split(regex_pattern, q_info['answer'])
            gold_answer = list([a.strip() for a in raw_gold_ans])
            if len(gold_answer) > 1:
                assert "What is the completion time" in q_info["query"] or "What is the start time" in q_info["query"]

            my_answer = a_info["answer"]

            if civic_is_correct_answer(my_answer=my_answer, gold_answer=gold_answer):
                correct_answer_num += 1

        m_config_to_correct_answer_num[config] = correct_answer_num

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/civic-answer-num.json", 'w') as file:
        json.dump({"num": m_config_to_correct_answer_num}, file, indent=4)

def civic2_eval_answer_recall_precicision_f1(configs):
    '''
    Recall, precision, f1 is defined as:
    https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    '''

    def civic2_normalize_project(s: str):
        pattern = r"\(.*?\)"
        new_s = re.sub(pattern, '', s)
        return white_space_fix(remove_punc(lower(normalize_string(new_s))))

    
    
    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/civic2.json", 'r') as file:
        queries = json.load(file)

    with open("/mnt/f/Research-2024-spring/RetrieveSys/data/attributes/civic/truths.json", 'r') as file:
        truths = json.load(file)

    q_len = 38

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/civic2.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_correct_answer_stat_recall = dict()
    m_config_to_correct_answer_stat_precision = dict()
    m_config_to_correct_answer_stat_f1 = dict()
    for config in configs:
        correct_answer_stat = {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0
        }
        my_answers = m_config_to_answers[config]
        assert len(my_answers) == len(queries)
        assert len(queries) == q_len
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])

            file_name = q_info["file_name"]
            projects_list = list([civic2_normalize_project(p) for p in truths[file_name].keys()])

            gold_answer = set([civic2_normalize_project(p) for p in q_info["answer"]])
            
            raw_raw_my_answer = a_info["answer"].split(",")
            raw_my_answer = [civic2_normalize_project(p) for p in raw_raw_my_answer]
            my_answer = set([p for p in raw_my_answer if p in projects_list])
            print(q_info["id"], config, my_answer, a_info["answer"])
            # assert len(my_answer) == len(raw_my_answer)

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

            correct_answer_stat["recall"] += recall
            correct_answer_stat["precision"] += precision
            correct_answer_stat["f1"] += f1

        m_config_to_correct_answer_stat_recall[config] = correct_answer_stat["recall"] / q_len
        m_config_to_correct_answer_stat_precision[config] = correct_answer_stat["precision"] / q_len
        m_config_to_correct_answer_stat_f1[config] = correct_answer_stat["f1"] / q_len

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/civic2-answer-recall-precision-f1.json", 'w') as file:
        json.dump({
            "avg_recall": m_config_to_correct_answer_stat_recall,
            "avg_precision": m_config_to_correct_answer_stat_precision,
            "avg_f1": m_config_to_correct_answer_stat_f1,
        }, file, indent=4)

def contract_eval_answer_accuracy(configs):
    def contract_normalize_answer(s: str):
        new_s = global_normalize_answer(s)
        choices = ["entailment", "contradiction", "notmentioned", "not mentioned"]
        ans = [c for c in choices if c in new_s]
        assert len(ans) <= 1
        if len(ans) == 0:
            return "none"
        if ans[0] == "not mentioned":
            return "notmentioned"
        else:
            return ans[0]

    def contract_is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        assert my_answer in ["entailment", "contradiction", "notmentioned", "none"]
        assert gold_answer in ["entailment", "contradiction", "notmentioned"]
        return my_answer == gold_answer

    

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/contract.json", 'r') as file:
        queries = json.load(file)

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/contract.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_correct_answer_num = dict()
    for config in configs:
        correct_answer_num = 0
        my_answers = m_config_to_answers[config]
        assert len(my_answers) == len(queries)
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])

            gold_answer = contract_normalize_answer(q_info["answer"])
            my_answer = contract_normalize_answer(a_info["answer"])

            if contract_is_correct_answer(my_answer=my_answer, gold_answer=gold_answer):
                correct_answer_num += 1

        m_config_to_correct_answer_num[config] = correct_answer_num

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/contract-answer-num.json", 'w') as file:
        json.dump({"num": m_config_to_correct_answer_num}, file, indent=4)

def contract_eval_answer_accuracy_llm_gold(configs):
    def contract_normalize_answer(s: str):
        new_s = global_normalize_answer(s)
        choices = ["entailment", "contradiction", "notmentioned", "not mentioned"]
        ans = [c for c in choices if c in new_s]
        assert len(ans) <= 1
        if len(ans) == 0:
            return "none"
        if ans[0] == "not mentioned":
            return "notmentioned"
        else:
            return ans[0]

    def contract_is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        assert my_answer in ["entailment", "contradiction", "notmentioned", "none"]
        assert gold_answer in ["entailment", "contradiction", "notmentioned", "none"]
        return my_answer == gold_answer

    

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/full/contract.json", 'r') as file:
        llm_gold_answers = json.load(file)

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/contract.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_correct_answer_num = dict()
    for config in configs:
        correct_answer_num = 0
        my_answers = m_config_to_answers[config]
        assert len(my_answers) == len(llm_gold_answers)
        for q_info, a_info in zip(llm_gold_answers, my_answers):
            assert q_info["qid"] == a_info["qid"]

            gold_answer = contract_normalize_answer(q_info["answer"])
            my_answer = contract_normalize_answer(a_info["answer"])

            if contract_is_correct_answer(my_answer=my_answer, gold_answer=gold_answer):
                correct_answer_num += 1

        m_config_to_correct_answer_num[config] = correct_answer_num

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/contract-answer-num-llm-gold.json", 'w') as file:
        json.dump({"num": m_config_to_correct_answer_num}, file, indent=4)

def qasper_eval_answer_f1(configs):
    from collections import Counter
    def qasper_normalize_answer(s: str):
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def qasper_token_f1_score(prediction: str, ground_truth: str) -> float:
        prediction_tokens = qasper_normalize_answer(prediction).split()
        ground_truth_tokens = qasper_normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
        

    

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'r') as file:
        queries = json.load(file)

    assert len(queries) == 1451

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/qasper.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_answer_f1 = dict()
    for config in configs:
        f1_score = 0
        my_answers = m_config_to_answers[config]
        assert len(my_answers) == len(queries)
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])
            gold_answers_list = q_info["answer"]
            my_answer = a_info["answer"]
            f1_score += max([qasper_token_f1_score(prediction=my_answer, ground_truth=a) for a in gold_answers_list])
        m_config_to_answer_f1[config] = f1_score / len(queries)

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/qasper-answer-f1.json", 'w') as file:
        json.dump({"avg_f1": m_config_to_answer_f1}, file, indent=4)


def qasper_eval_gold_answer_f1(configs):
    from collections import Counter
    def qasper_normalize_answer(s: str):
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def qasper_token_f1_score(prediction: str, ground_truth: str) -> float:
        prediction_tokens = qasper_normalize_answer(prediction).split()
        ground_truth_tokens = qasper_normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
        

    

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'r') as file:
        queries = json.load(file)

    assert len(queries) == 1451

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/qasper.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    m_config_to_answer_f1 = dict()
    for config in configs:
        f1_score = 0
        for q_info in queries:
            my_answers_list = [a["answer"] for a in m_config_to_answers[config] if a["qid"] == q_info["id"]]
            gold_answers_list = q_info["answer"]
            assert len(gold_answers_list) == len(my_answers_list)
            assert [a["aid"] for a in m_config_to_answers[config] if a["qid"] == q_info["id"]] == list(range(len(gold_answers_list)))
            f1_score += max([qasper_token_f1_score(prediction=ma, ground_truth=ga) for ma, ga in zip(my_answers_list, gold_answers_list)])
        m_config_to_answer_f1[config] = f1_score / len(queries)

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/qasper-answer-f1-gold.json", 'w') as file:
        json.dump({"avg_f1": m_config_to_answer_f1}, file, indent=4)

if __name__ == "__main__":
    configs = ["bm25", "bm25-o", "dpr", "dpr-o", "sbert", "sbert-o", "raptor", "raptor-o", "sht", "sht-r", "sht-g", "sht-r-g", "sht-abs", "sht-r-abs", "sht-g-abs", "sht-r-g-abs", "sht-bm25", "sht-r-bm25", "sht-bm25-g", "sht-r-bm25-g", "sht-bm25-abs", "sht-r-bm25-abs", "sht-bm25-g-abs", "sht-r-bm25-g-abs", "sht-dpr", "sht-r-dpr", "sht-dpr-g", "sht-r-dpr-g", "sht-dpr-abs", "sht-r-dpr-abs", "sht-dpr-g-abs", "sht-r-dpr-g-abs", "te3small", "te3small-o", "sht-te3small", "sht-r-te3small", "sht-te3small-g", "sht-r-te3small-g", "sht-te3small-abs", "sht-r-te3small-abs", "sht-te3small-g-abs", "sht-r-te3small-g-abs"]
    # configs = ["gold", "gold-short"]
    # civic_eval_answer_accuracy(configs)
    # civic2_eval_answer_recall_precicision_f1(configs)
    # contract_eval_answer_accuracy(configs)
    # qasper_eval_answer_f1(configs)
    # qasper_eval_gold_answer_f1(configs)
    contract_eval_answer_accuracy_llm_gold(configs)


        
