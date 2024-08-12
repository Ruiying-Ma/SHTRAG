import json
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
from sht import split_text_into_sentences

def gen_qasper_queries():
    with open("/mnt/f/Research-2024-spring/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json", 'r') as file:
        data = json.load(file)
    
    queries = []
    prompt_tmpl_1 = "Please answer the question below using the provided context in one sentence. If the context does not contain enough information to answer the question, respond with 'Unanswerable'.\n\n[Begin of Question]\n{query}\n[End of Question]\n\n"
    prompt_tmpl_2 = "[Begin of Context]\n{context}\n[End of Context]\n\nAnswer (in one sentence): "
    for doc in data:
        doc_info = data[doc]
        file_name = doc
        for query_info in doc_info["qas"]:
            query = query_info["question"]
            answers = []
            contexts = []
            short_contexts = []
            types = []
            for annotation_info in query_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    answer = "Unanswerable"
                    context = []
                    short_context = []
                    type = "none"
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    
                    evidence = "\n".join([text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text])
                    if len(evidence) == 0:
                        context = []
                    else:
                        context = split_text_into_sentences([".", "!", "?", "\n"], evidence)

                    short_evidence = "\n".join([text for text in answer_info["highlighted_evidence"] if "FLOAT SELECTED" not in text])
                    if len(short_evidence) == 0:
                        short_context = []
                    else:
                        short_context = split_text_into_sentences([".", "!", "?", "\n"], short_evidence)
                
                answers.append(answer)
                contexts.append(context)
                short_contexts.append(short_context)
                types.append(type)
            
            id = len(queries)
            queries.append({
                "id": id,
                "file_name": file_name,
                "query": query,
                "prompt_template": prompt_tmpl_1.format(query=query) + prompt_tmpl_2,
                "answer": answers,
                "context": contexts,
                "short_context": short_contexts,
                "type": types,
            })

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/qasper.json", 'w') as file:
        json.dump(queries, file, indent=4)

            
if __name__ == "__main__":
    gen_qasper_queries()                   
                    