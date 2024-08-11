import json
import os


def extract_answers_from_batch(batch_result_file):
    answers = dict()
    with open(batch_result_file, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            gpt_response = json_line["response"]["body"]["choices"][0]["message"]["content"]
            custom_id = json_line["custom_id"]
            tags = custom_id.split(":")
            config = tags[0]
            dataset = tags[1]
            query_id = tags[2]

            if config not in answers:
                answers[config] = dict()

            if dataset not in answers[config]:
                answers[config][dataset] = []

            answers[config][dataset].append({
                "qid": query_id,
                "answer": gpt_response,
            })

    for config in answers:
        for dataset in answers[config]:
            assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json")
            with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json", 'w') as file:
                json.dump(answers[config][dataset], file, indent=4)


def extract_gold_answers_from_batch_contract(batch_result_file):
    answers = dict()
    with open(batch_result_file, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            gpt_response = json_line["response"]["body"]["choices"][0]["message"]["content"]
            custom_id = json_line["custom_id"]
            query_id = int(custom_id)
            config = "gold"
            dataset = "contract"

            if config not in answers:
                answers[config] = dict()

            if dataset not in answers[config]:
                answers[config][dataset] = []

            answers[config][dataset].append({
                "qid": query_id,
                "answer": gpt_response,
            })

    for config in answers:
        for dataset in answers[config]:
            assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json")
            with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json", 'w') as file:
                json.dump(answers[config][dataset], file, indent=4)


def extract_gold_answers_from_batch_qasper(batch_result_file):
    answers = dict()
    with open(batch_result_file, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            gpt_response = json_line["response"]["body"]["choices"][0]["message"]["content"]
            custom_id = json_line["custom_id"]
            tags = custom_id.split(":")
            query_id = int(tags[0])
            answer_id = int(tags[1])
            context_type = tags[2]
            if context_type == "long":
                config = "gold"
            else:
                config = "gold-short"
            dataset = "qasper"

            if config not in answers:
                answers[config] = dict()

            if dataset not in answers[config]:
                answers[config][dataset] = []

            answers[config][dataset].append({
                "qid": query_id,
                "aid": answer_id,
                "answer": gpt_response,
            })

    for config in answers:
        for dataset in answers[config]:
            assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json")
            with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/{dataset}.json", 'w') as file:
                json.dump(answers[config][dataset], file, indent=4)

if __name__ == "__main__":
    batch_result_file = "/mnt/f/Research-2024-spring/SHTRAG/batches/contract-full-context-results.jsonl"
    extract_answers_from_batch(batch_result_file)