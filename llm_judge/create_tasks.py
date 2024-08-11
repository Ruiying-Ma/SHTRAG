import json
import os
import tiktoken

def create_llm_judge_tasks_for_qasper(model="gpt-4o-mini", max_tokens=100):
    with open("/mnt/f/Research-2024-spring/SHTRAG/llm_judge/prompt.txt", 'r') as file:
        prompt_template = file.read()

    # configs = ["sht", "sht-r", "sht-g", "sht-r-g", "raptor", "bm25", "sbert", "dpr"]
    # configs = ["raptor-o", "bm25-o", "sbert-o", "dpr-o"]

    # configs = ["bm25", "bm25-o", "dpr", "dpr-o", "sbert", "sbert-o", "raptor", "raptor-o"]
    # configs = ["sht", "sht-r", "sht-g", "sht-r-g", "sht-abs", "sht-r-abs", "sht-g-abs", "sht-r-g-abs"] 
    # configs = ["sht-bm25", "sht-r-bm25", "sht-bm25-g", "sht-r-bm25-g", "sht-bm25-abs", "sht-r-bm25-abs", "sht-bm25-g-abs", "sht-r-bm25-g-abs"]
    # configs = ["sht-dpr", "sht-r-dpr", "sht-dpr-g", "sht-r-dpr-g", "sht-dpr-abs", "sht-r-dpr-abs", "sht-dpr-g-abs", "sht-r-dpr-g-abs"]
    # configs = ["te3small", "te3small-o"]
    configs = ["sht-te3small", "sht-r-te3small", "sht-te3small-g", "sht-r-te3small-g", "sht-te3small-abs", "sht-r-te3small-abs", "sht-te3small-g-abs", "sht-r-te3small-g-abs"]


    tokenizer = tiktoken.get_encoding("cl100k_base")

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'r') as file:
        gold_data = json.load(file)

    json_lines = []
    custom_id_set = set()
    token_count = 0
    prompt_num = sum([len(gd["answer"]) for gd in gold_data])

    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/qasper.json", 'r') as file:
            my_answers_list = json.load(file)

        assert len(my_answers_list) == len(gold_data)

        for my_answer, gold_qa_info in zip(my_answers_list, gold_data):
            assert int(my_answer["qid"]) == gold_qa_info["id"]
            query_id = gold_qa_info["id"]
            for gold_answer_id, gold_answer in enumerate(gold_qa_info["answer"]):
                custom_id = config + ":" + "qasper" + ":" + str(query_id) + ":" + str(gold_answer_id)
                assert custom_id not in custom_id_set
                custom_id_set.add(custom_id)
                user_prompt = prompt_template.format(query=query_id, reference=gold_answer, answer=my_answer["answer"])

                json_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": user_prompt,
                        }],
                        "max_tokens": max_tokens,
                    }
                })

                token_count += len(tokenizer.encode(user_prompt))

    assert len(json_lines) == prompt_num * len(configs)

    # with open("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge.jsonl", 'w') as file:
    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge-sht-te3small-abs.jsonl")
    with open("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge-sht-te3small-abs.jsonl", 'w') as file:
        for id, json_line in enumerate(json_lines):
            json.dump(json_line, file)
            if id != len(json_lines) - 1:
                file.write("\n")

    print(token_count)

def create_llm_judge_tasks_for_qasper_gold(model="gpt-4o-mini", max_tokens=100):
    with open("/mnt/f/Research-2024-spring/SHTRAG/llm_judge/prompt.txt", 'r') as file:
        prompt_template = file.read()

    configs = ["gold", "gold-short"]


    tokenizer = tiktoken.get_encoding("cl100k_base")

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'r') as file:
        gold_data = json.load(file)

    json_lines = []
    custom_id_set = set()
    token_count = 0
    prompt_num = sum([len(gd["answer"]) for gd in gold_data])

    for config in configs:
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/answers/{config}/qasper.json", 'r') as file:
            my_data = json.load(file)

        for gold_qa_info in gold_data:
            query_id = gold_qa_info["id"]
            my_answers_list = [a["answer"] for a in my_data if a["qid"] == query_id]
            assert len(my_answers_list) == len(gold_qa_info["answer"])
            assert [a["aid"] for a in my_data if a["qid"] == query_id] == list(range(len(gold_qa_info["answer"])))
            for gold_answer_id, gold_answer in enumerate(gold_qa_info["answer"]):
                custom_id = config + ":" + "qasper" + ":" + str(query_id) + ":" + str(gold_answer_id)
                assert custom_id not in custom_id_set
                custom_id_set.add(custom_id)
                user_prompt = prompt_template.format(query=query_id, reference=gold_answer, answer=my_answers_list[gold_answer_id])

                json_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": user_prompt,
                        }],
                        "max_tokens": max_tokens,
                    }
                })

                token_count += len(tokenizer.encode(user_prompt))

    assert len(json_lines) == prompt_num * len(configs)

    # with open("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge.jsonl", 'w') as file:
    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge-gold.jsonl")
    with open("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper-llm-judge-gold.jsonl", 'w') as file:
        for id, json_line in enumerate(json_lines):
            json.dump(json_line, file)
            if id != len(json_lines) - 1:
                file.write("\n")

    print(token_count)


if __name__ == "__main__":
    create_llm_judge_tasks_for_qasper_gold()