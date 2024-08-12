import json
import os

def gen_contract_queries():
    with open("/mnt/f/Research-2024-spring/contract-nli/contract-nli/test.json", 'r') as file:
        data = json.load(file)

    queries = []

    query_tmpl = "{descrip}: {hypo}"

    prompt_tmpl_1 = "Please determine whether the given contract excerpt entails, contradicts, or does not mention the given hypothesis. Use one of the following labels: 'Entailment,' 'Contradiction,' or 'Not Mentioned.'\n\n[Begin of Hypothesis]\n{descrip}: {hypo}\n[End of Hypothesis]\n\n[Begin of Contract Excerpt]"

    prompt_tmpl_2 = '''\n{context}\n[End of Contract Excerpt]\n\nAnswer: [Entailment / Contradiction / Not Mentioned]'''

    json_names = os.listdir("/mnt/f/Research-2024-spring/RetrieveSys/data/adobe_extract/contract")

    for document in data["documents"]:
        file_name = document["file_name"].replace(".pdf", "").strip()
        json_name = file_name + ".json"
        if not json_name in json_names:
            continue
        assert len(document["annotation_sets"]) == 1
        annotation_dict = document["annotation_sets"][0]["annotations"]
        print(len(annotation_dict))
        for q_key in annotation_dict:
            q = data["labels"][q_key]
            id = len(queries)
            span_ids = annotation_dict[q_key]["spans"]
            sentences = []
            for span_id in span_ids:
                span = document["spans"][span_id]
                sentences.append(document["text"][span[0]:(span[1] + 1)])

            queries.append({
                "id": id,
                "file_name": file_name,
                "query": query_tmpl.format(descrip=q["short_description"], hypo=q["hypothesis"]),
                "prompt_template": prompt_tmpl_1.format(descrip=q["short_description"], hypo=q["hypothesis"]) + prompt_tmpl_2,
                "answer": annotation_dict[q_key]["choice"],
                "context": sentences,
            })

    pdf_names = [doc["file_name"].strip() for doc in data["documents"]]
    for json_name in json_names:
        pdf_name = json_name.replace(".json", ".pdf")
        if not (pdf_name in pdf_names):
            print(json_name)

    print(pdf_names)

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/contract.json", 'w') as file:
        json.dump(queries, file, indent=4)

if __name__ == "__main__":
    gen_contract_queries()

