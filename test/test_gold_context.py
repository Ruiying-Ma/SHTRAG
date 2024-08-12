import json
import os
import token

def create_contract_batch_gold_context(max_tokens=100, model="gpt-4o-mini"):
    context_lists = []
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/contract.json", 'r') as file:
        queries = json.load(file)

    for qa_info in queries:
        context = ""
        for c in qa_info["context"]:
            if c.endswith("\n"):
                context += c
            else:
                context += c + "\n"
        context_lists.append({
            "id": qa_info["id"],
            "prompt_template": qa_info["prompt_template"],
            "context": context
        })

    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/contexts/gold/contract.json")
    with open("/mnt/f/Research-2024-spring/SHTRAG/contexts/gold/contract.json", 'w') as file:
        json.dump(context_lists, file, indent=4)

def create_qasper_batch_gold_context_long_short(max_tokens=100, model="gpt-4o-mini"):
    json_lines = []
    with open("/mnt/f/Research-2024-spring/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json", 'r') as file:
        data = json.load(file)

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'r') as file:
        queries = json.load(file)

    contexts = []
    short_contexts = []
    for doc in data:
        doc_info = data[doc]
        for query_info in doc_info["qas"]:
            cur_contexts = []
            cur_short_contexts = []
            for annotation_info in query_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    evidence = ""
                    short_evidence = ""
                else:
                    evidence = "\n".join([text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text])
                    short_evidence = "\n".join([text for text in answer_info["highlighted_evidence"] if "FLOAT SELECTED" not in text])
                
                cur_contexts.append(evidence)
                cur_short_contexts.append(short_evidence)
            contexts.append(cur_contexts)
            short_contexts.append(cur_short_contexts)
    assert len(contexts)== len(queries)
    assert len(short_contexts) == len(queries)
    assert all([len(c) == len(q["context"]) for c, q in zip(contexts, queries)])
    assert all([len(c) == len(q["short_context"]) for c, q in zip(short_contexts, queries)])

    for qa_info, context, short_context in zip(queries, contexts, short_contexts):
        assert len(context) == len(short_context)
        for id, c in enumerate(context):
            sc = short_context[id]
            custom_id_context = str(qa_info["id"]) + ":" + str(id) + ":long"
            custom_id_short_context = str(qa_info["id"]) + ":" + str(id) + ":short"
            prompt_context = qa_info["prompt_template"].format(context=c)
            prompt_short_context = qa_info["prompt_template"].format(context=sc)
            json_lines.append({
                "custom_id": custom_id_context,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt_context,
                    }],
                    "max_tokens": max_tokens,
                }
            })
            json_lines.append({
                "custom_id": custom_id_short_context,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt_short_context,
                    }],
                    "max_tokens": max_tokens,
                }
            })
    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper_gold_context.jsonl")
    with open("/mnt/f/Research-2024-spring/SHTRAG/batches/qasper_gold_context.jsonl", 'w') as file:
        for id, json_line in enumerate(json_lines):
            json.dump(json_line, file)
            if id != len(json_lines) - 1:
                file.write("\n")


def create_contract_batch_full_context():
    context_lists = []
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/contract.json", 'r') as file:
        queries = json.load(file)

    m_file_name_to_text = dict()
    with open("/mnt/f/Research-2024-spring/contract-nli/contract-nli/test.json", 'r') as file:
        data = json.load(file)
    for doc in data["documents"]:
        file_name = doc["file_name"].replace(".pdf", "").strip()
        m_file_name_to_text[file_name] = doc["text"]
    
    for qa_info in queries:
        assert qa_info["file_name"] in m_file_name_to_text
        context_lists.append({
            "id": qa_info["id"],
            "prompt_template": qa_info["prompt_template"],
            "context": m_file_name_to_text[qa_info["file_name"]]
        })

    assert not os.path.exists("/mnt/f/Research-2024-spring/SHTRAG/contexts/full/contract.json")
    with open("/mnt/f/Research-2024-spring/SHTRAG/contexts/full/contract.json", 'w') as file:
        json.dump(context_lists, file, indent=4)

if __name__ == "__main__":
    create_contract_batch_full_context()