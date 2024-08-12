import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
import json
import unicodedata
import re
import string
from typing import List
from sht import split_text_into_sentences
import tiktoken

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

def jaccard_sim(s1, s2):
    def jaccard_normalize_string(s):
        return white_space_fix(lower(normalize_string(s)))
    set1 = set(jaccard_normalize_string(s1).split())
    set2 = set(jaccard_normalize_string(s2).split())

    assert len(set1) > 0
    assert len(set2) > 0

    overlap = len(set1.intersection(set2))
    union = len(set1.union(set2))

    assert union > 0
    return overlap / union 

def comp_contexts(my_contexts, gold_contexts):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    scores = []
    tokens = [len(tokenizer.encode(gc)) for gc in gold_contexts]
    for gc in gold_contexts:
        score = max([jaccard_sim(mc, gc) for mc in my_contexts])
        scores.append(score)
    
    weighted_score = sum([s * t for s, t in zip(scores, tokens)]) / sum(tokens)
    return weighted_score


def civic_wrong_answers(configs):

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

    for config in configs:
        correct_answer_num = 0
        my_answers = m_config_to_answers[config]
        wrong_answers = []
        context_path = f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/civic-1000.json"
        with open(context_path, 'r') as file:
            context_info = json.load(file)
        assert len(my_answers) == len(queries)
        assert len(my_answers) == len(context_info)
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
            else:
                context = context_info[q_info["id"]]
                assert context["id"] == q_info["id"]
                wrong_answers.append({
                    "id": q_info["id"],
                    "file_name": q_info["file_name"],
                    "query": q_info["query"],
                    "my_answer": a_info["answer"],
                    "gold_answer": q_info["answer"],
                    "context": context["context"],
                })
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/wrong_answers/civic/{config}.json", 'w') as file:
            json.dump(wrong_answers, file, indent=4)

def contract_wrong_answers(configs):
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

    for config in configs:
        correct_answer_num = 0
        my_answers = m_config_to_answers[config]
        wrong_answers = []
        context_path = f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/contract-1000.json"
        with open(context_path, 'r') as file:
            context_info = json.load(file)
        assert len(my_answers) == len(queries)
        assert len(my_answers) == len(context_info)
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])

            gold_answer = contract_normalize_answer(q_info["answer"])
            my_answer = contract_normalize_answer(a_info["answer"])

            if contract_is_correct_answer(my_answer=my_answer, gold_answer=gold_answer):
                correct_answer_num += 1
            else:
                context = context_info[q_info["id"]]
                assert context["id"] == q_info["id"]
                wrong_answers.append({
                    "id": q_info["id"],
                    "file_name": q_info["file_name"],
                    "query": q_info["query"],
                    "my_answer": a_info["answer"],
                    "gold_answer": q_info["answer"],
                    "context": context["context"],
                    "gold_context": q_info["context"],
                })

        with open(f"/mnt/f/Research-2024-spring/SHTRAG/wrong_answers/contract/{config}.json", 'w') as file:
            json.dump(wrong_answers, file, indent=4)


def contract_answers_and_contexts(configs):
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

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/contract.json", 'r') as file:
        queries = json.load(file)

    m_config_to_answers = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/contract.json", 'r') as file:
            data = json.load(file)
        m_config_to_answers[config] = data

    for config in configs:
        my_answers = m_config_to_answers[config]
        answer_stat = dict() # my_answer, gold_answer
        context_stat = dict() # my_answer, gold_answer
        context_path = f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/contract-1000.json"
        with open(context_path, 'r') as file:
            context_info = json.load(file)
        assert len(my_answers) == len(queries)
        assert len(my_answers) == len(context_info)
        for q_info, a_info in zip(queries, my_answers):
            assert q_info["id"] == int(a_info["qid"])

            gold_answer = contract_normalize_answer(q_info["answer"])
            my_answer = contract_normalize_answer(a_info["answer"])

            if my_answer not in answer_stat:
                answer_stat[my_answer] = dict()
            if my_answer not in context_stat:
                context_stat[my_answer] = dict()
            if gold_answer not in answer_stat[my_answer]:
                answer_stat[my_answer][gold_answer] = 0
            if gold_answer not in context_stat[my_answer]:
                context_stat[my_answer][gold_answer] = 0.0

            answer_stat[my_answer][gold_answer] += 1

            my_context = context_info[q_info["id"]]["context"]
            gold_contexts = q_info["context"]
            if len(gold_contexts) > 0:
                assert gold_answer != "notmentioned"
                context_stat[my_answer][gold_answer] += comp_contexts(my_contexts=split_text_into_sentences([".", "!", "?", "\n"], my_context), gold_contexts=gold_contexts)
            else:
                assert gold_answer == "notmentioned"

        for a1 in ["entailment", "contradiction", "notmentioned", "none"]:
            for a2 in ["entailment", "contradiction", "notmentioned"]:
                if a1 in answer_stat and a2 in answer_stat[a1]:
                    if answer_stat[a1][a2] != 0:
                        context_stat[a1][a2] /= answer_stat[a1][a2]

        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answer_context_stats/contract/{config}.json", 'w') as file:
            json.dump({
                "answers": answer_stat,
                "contexts": context_stat,
            }, file, indent=4)

if __name__ == "__main__":
    # configs = ["sht", "sht-r", "sht-g", "sht-r-g"]
    configs = ["full"]
    # contract_wrong_answers(configs)
    # civic_wrong_answers(configs)
    contract_answers_and_contexts(configs)