import json
import os

def extract_ratings_from_batch(batch_result_files):
    ratings = dict()
    for batch_result_file in batch_result_files:
        with open(batch_result_file, 'r') as file:
            for line in file:
                json_line = json.loads(line.strip())
                gpt_response = json_line["response"]["body"]["choices"][0]["message"]["content"]
                custom_id = json_line["custom_id"]
                tags = custom_id.split(":")
                config = tags[0]
                assert tags[1] == "qasper"
                query_id = tags[2]
                answer_id = tags[3]

                if config not in ratings:
                    ratings[config] = []
                if int(query_id) == len(ratings[config]):
                    assert answer_id == "0"
                    ratings[config].append({
                        "qid": int(query_id),
                        "rating": [gpt_response],
                    })

                else:
                    assert int(query_id) == len(ratings[config]) - 1
                    assert int(query_id) == ratings[config][-1]["qid"]
                    assert int(answer_id) == len(ratings[config][-1]["rating"])
                    assert int(answer_id) > 0
                    ratings[config][-1]["rating"].append(gpt_response)

    for config in ratings:
        assert sum([len(r["rating"]) for r in ratings[config]]) == 3554

    # assert len(ratings) == 32
    # assert set(ratings.keys()) == set(["bm25", "bm25-o", "dpr", "dpr-o", "sbert", "sbert-o", "raptor", "raptor-o", "sht", "sht-r", "sht-g", "sht-r-g", "sht-abs", "sht-r-abs", "sht-g-abs", "sht-r-g-abs", "sht-bm25", "sht-r-bm25", "sht-bm25-g", "sht-r-bm25-g", "sht-bm25-abs", "sht-r-bm25-abs", "sht-bm25-g-abs", "sht-r-bm25-g-abs", "sht-dpr", "sht-r-dpr", "sht-dpr-g", "sht-r-dpr-g", "sht-dpr-abs", "sht-r-dpr-abs", "sht-dpr-g-abs", "sht-r-dpr-g-abs"])

    for config in ratings:
        assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/llm_judge/qasper/{config}.json")
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/llm_judge/qasper/{config}.json", 'w') as file:
            json.dump(ratings[config], file, indent=4)


if __name__ == "__main__":
    batch_result_files = ["/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge-gold-results.jsonl"]
    extract_ratings_from_batch(batch_result_files)