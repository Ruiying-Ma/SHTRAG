from utils import *
from collections import Counter

def qasper_eval_answer_f1(
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
    def normalize_answer(s: str):
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def token_f1(my_answer: str, gold_answer: str) -> float:
        my_answer_tokens = normalize_answer(my_answer).split()
        gold_answer_tokens = normalize_answer(gold_answer).split()
        common = Counter(my_answer_tokens) & Counter(gold_answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(my_answer_tokens)
        recall = 1.0 * num_same / len(gold_answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    gold_answers = get_gold_answers("qasper")
    my_answers = get_answers(
        "qasper",
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

    assert len(my_answers) == len(gold_answers)
    assert len(gold_answers) == 1451
    
    f1 = 0.0
    n_answer = 0
    for gold_answer, my_answer in zip(gold_answers, my_answers):
        assert gold_answer["id"] == my_answer["id"]
        if len(gold_answer['answer']) == 0:
            continue
        f1 += max([token_f1(my_answer=my_answer["answer"], gold_answer=ga) for ga in gold_answer["answer"]])
        n_answer += 1

    print(f"Qasper: among {n_answer} queries\n\tf1={round(f1  * 100 / n_answer, 3)}")

    return round(f1 * 100 / n_answer, 3)


def qasper_eval_answer_llm(
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
    def extract_rating_from_gpt_response(s: str):
        match = re.search(r"Rating:\s*\[\[(\d)\]\]", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*\[(\d)\]", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*(\d)", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*\*\*(\d)\*\*", s)
        if match:
            return int(match.group(1))
        raise ValueError(f"'{s}' is the wrong form...")

    ratings = get_ratings(
        "qasper",
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

    assert len(ratings) == 1451
    
    tot_rating = 0.0
    n_answer = 0
    for rating_info in ratings:
        candid_ratings = [extract_rating_from_gpt_response(r) for r in rating_info["rating"]]
        if len(candid_ratings) == 0:
            continue
        rating = max(candid_ratings)
        tot_rating += rating
        n_answer += 1


    print(f"Qasper: among {n_answer} queries\n\trating={round(tot_rating * 100 / (3 * n_answer), 3)}")

    return round(tot_rating * 100 / (3 * n_answer), 3)

if __name__ == "__main__":
    embedding_model = "te3small"
    qasper_eval_answer_llm(
        chunk_size=100,
        summary_len=100,
        node_embedding_model=embedding_model,
        query_embedding_model=embedding_model,
        summarization_model="gpt-4o-mini",
        embed_hierarchy=True,
        distance_metric="cosine",
        context_hierarchy=True,
        context_raw=True,
        context_len=1000,
        is_intrinsic=False,
        is_baseline=True,
        is_raptor=False,
        is_ordered=False,
        is_grobid=False
    )
    