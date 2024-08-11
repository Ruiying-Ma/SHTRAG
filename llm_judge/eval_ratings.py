import json
import re

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

def qasper_eval_answer_score():
    configs = ["bm25", "bm25-o", "dpr", "dpr-o", "sbert", "sbert-o", "raptor", "raptor-o", "sht", "sht-r", "sht-g", "sht-r-g", "sht-abs", "sht-r-abs", "sht-g-abs", "sht-r-g-abs", "sht-bm25", "sht-r-bm25", "sht-bm25-g", "sht-r-bm25-g", "sht-bm25-abs", "sht-r-bm25-abs", "sht-bm25-g-abs", "sht-r-bm25-g-abs", "sht-dpr", "sht-r-dpr", "sht-dpr-g", "sht-r-dpr-g", "sht-dpr-abs", "sht-r-dpr-abs", "sht-dpr-g-abs", "sht-r-dpr-g-abs", "te3small", "te3small-o", "sht-te3small", "sht-r-te3small", "sht-te3small-g", "sht-r-te3small-g", "sht-te3small-abs", "sht-r-te3small-abs", "sht-te3small-g-abs", "sht-r-te3small-g-abs", "gold", "gold-short"]

    m_config_to_answer_scores = dict()
    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/llm_judge/qasper/{config}.json", 'r') as file:
            data = json.load(file)
        assert len(data) == 1451
        m_config_to_answer_scores[config] = data


    m_config_to_answer_llm_rating = dict()
    for config in configs:
        rating_sum = []
        my_answers = m_config_to_answer_scores[config]
        assert len(my_answers) == 1451
        count = 0
        for a_info in my_answers:
            my_ratings = a_info["rating"]
            rating_sum.append(max([extract_rating_from_gpt_response(a) for a in my_ratings]))
            count += len(my_ratings)
        assert count == 3554
        assert len(rating_sum) == 1451
        m_config_to_answer_llm_rating[config] = sum(rating_sum) / len(my_answers)

    with open("/mnt/f/Research-2024-spring/SHTRAG/answers/qasper-answer-llm-rating.json", 'w') as file:
        json.dump({"avg_rating": m_config_to_answer_llm_rating}, file, indent=4)

if __name__ == "__main__":
    qasper_eval_answer_score()