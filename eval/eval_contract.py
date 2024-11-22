from utils import *

def contract_eval_answer(
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
    Evaluate answer accuracy for contract-nli. 

    Filter out queries whose gold answers are NotMentioned.
    '''
    def normalize_answer(s: str):
        new_s = global_normalize_answer(s)
        choices = ["entailment", "contradiction"]
        ans = [c for c in choices if c in new_s]
        assert len(ans) <= 1
        if len(ans) == 0:
            return ""
        else:
            return ans[0]

    def is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        assert my_answer in ["entailment", "contradiction", ""]
        assert gold_answer in ["entailment", "contradiction"]
        return my_answer == gold_answer

    gold_answers = get_gold_answers("contract")

    my_answers = get_answers(
        "contract",
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

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 1241

    n_correct = 0
    n_ans = 0
    for gold_answer, my_answer in zip(gold_answers, my_answers):
        assert gold_answer["id"] == my_answer["id"]
        if gold_answer["answer"] == "NotMentioned":
            continue

        if is_correct_answer(my_answer=normalize_answer(my_answer["answer"]), gold_answer=normalize_answer(gold_answer["answer"])):
            n_correct += 1
        n_ans += 1

    assert n_ans == 669

    print(f"ContractNLI: {n_correct} correct answers among {n_ans} queries\n\taccuracy={round(n_correct * 100 / n_ans, 3)}")

    return round(n_correct * 100 / n_ans, 3)

if __name__ == "__main__":
    contract_eval_answer(
        chunk_size=100,
        summary_len=100,
        node_embedding_model="te3small",
        query_embedding_model="te3small",
        summarization_model="gpt-4o-mini",
        embed_hierarchy=True,
        distance_metric="cosine",
        context_hierarchy=True,
        context_raw=True,
        context_len=1000,
        is_intrinsic=False,
        is_baseline=True,
        is_raptor=True,
        is_ordered=True,
        is_grobid=False
    )