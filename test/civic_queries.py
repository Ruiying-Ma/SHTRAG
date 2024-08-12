import json
import random

def gen_civic_query_2():
    query_template = "Identify and list the names of all projects that are in the specified status '{status}' and simultaneously of the type '{type}'."

    prompt_template = " Please refer only to the provided context and ensure that the project names are listed exactly as they appear. If no projects meet both criteria, return an empty list.\n\n[Begin of Context]\n{context}\n[End of Context]\n\nAnswer: [list the project names here]"

    with open("/mnt/f/Research-2024-spring/RetrieveSys/data/attributes/civic/truths.json", 'r') as file:
        data = json.load(file)

    queries = []

    for file_name in data:
        proj_lists = list(data[file_name].keys())
        
        status_set = set(list([data[file_name][p]["status"] for p in proj_lists]))
        type_set = set(list([data[file_name][p]["type"] for p in proj_lists]))

        assert len(status_set) * len(type_set) >= 2

        s1 = random.choice(list(status_set))
        s2 = random.choice(list(status_set))
        t1 = random.choice(list(type_set))
        t2 = random.choice(list(type_set))

        id = len(queries)
        query = query_template.format(status=s1, type=t1)
        answer = [p for p in proj_lists if (data[file_name][p]["status"] == s1 and data[file_name][p]["type"] == t1)]
        prompt_tmpl = query + prompt_template
        queries.append({
            "id": id,
            "file_name": file_name,
            "query": query,
            "prompt_template": prompt_tmpl,
            "answer": answer,
            "context": [],
        })

        id = len(queries)
        query = query_template.format(status=s2, type=t2)
        answer = [p for p in proj_lists if (data[file_name][p]["status"] == s2 and data[file_name][p]["type"] == t2)]
        prompt_tmpl = query + prompt_template
        queries.append({
            "id": id,
            "file_name": file_name,
            "query": query,
            "prompt_template": prompt_tmpl,
            "answer": answer,
            "context": [],
        })


            
    assert [q["id"] for q in queries] == list(range(len(queries)))
    assert len(queries) == len(data) * 2

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/civic2.json", 'w') as file:
        json.dump(queries, file, indent=4)


def gen_civic_query():
    '''
    For each query template, sample 5 projects. For each st/et_tmpl, choose 3 projects with answer, and 2 projects with none as answer.

    query_templates: for retrieval

    prompt_templates: for answering
    '''
    
    query_templates = {
        "status": "What is the status of the project '{proj}'?",
        "type": "What is the type of the project '{proj}'?",
        "st": "What is the start time of the project '{proj}'?",
        "et": "What is the completion time of the project '{proj}'?",
    }

    prompt_templates = {
        "status": " Please select one of the following: 'not started', 'construction', 'design', 'completed'.\n\n[Begin of Context]\n{context}\n[End of Context]\n\nAnswer: [not started / construction / design / completed]",
        "type": " Please respond with 'disaster' or 'capital'.\n\n[Begin of Context]\n{context}\n[End of Context]\n\nAnswer: [disaster / capital]",
        "st": " If the start time is not provided, please respond with 'none'.\n\n[Begin of Context]\n{context}\n[End of Context]\n\nAnswer: [start time / none]",
        "et": " If the completion time is not provided, please respond with 'none'.\n\n[Begin of Context]\n{context}\n[End of Context]\n\nAnswer: [completion time / none]",
    }

    with open("/mnt/f/Research-2024-spring/RetrieveSys/data/attributes/civic/truths.json", 'r') as file:
        data = json.load(file)

    queries = []

    for file_name in data:
        proj_lists = list(data[file_name].keys())
        status_proj_lists = random.sample(proj_lists, 5)
        type_proj_lists = random.sample(proj_lists, 5)

        proj_lists_has_st = [p for p in proj_lists if data[file_name][p]["st"].lower() != "none"]
        proj_lists_no_st = [p for p in proj_lists if data[file_name][p]["st"].lower() == "none"]
        proj_lists_has_et = [p for p in proj_lists if data[file_name][p]["et"].lower() != "none"]
        proj_lists_no_et = [p for p in proj_lists if data[file_name][p]["et"].lower() == "none"]

        assert len(proj_lists_has_st) >= 3 and len(proj_lists_has_et) >= 3

        st_proj_lists = random.sample(proj_lists_has_st, 3) + random.sample(proj_lists_no_st, 2)
        et_proj_lists = random.sample(proj_lists_has_et, 3) + random.sample(proj_lists_no_et, 2)

        projects = {
            "status": status_proj_lists,
            "type": type_proj_lists,
            "st": st_proj_lists,
            "et": et_proj_lists,
        }

        for key in query_templates:
            for proj_id in range(5):
                id = len(queries)
                query = query_templates[key].format(proj=projects[key][proj_id])
                answer = data[file_name][projects[key][proj_id]][key]
                prompt_tmpl = query + prompt_templates[key]
                queries.append({
                    "id": id,
                    "file_name": file_name,
                    "query": query,
                    "prompt_template": prompt_tmpl,
                    "answer": answer,
                    "context": [],
                })
    assert [q["id"] for q in queries] == list(range(len(queries)))
    assert len(queries) == len(data) * len(query_templates) * 5

    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/civic.json", 'w') as file:
        json.dump(queries, file, indent=4)

def handle_st_et():
    '''
    Split by /, -
    '''
    with open("/mnt/f/Research-2024-spring/SHTRAG/queries/civic.json", 'r') as file:
        queries = json.load(file)

    for qa_info in queries:
        if "time" in qa_info["query"] and qa_info["answer"] != "none":
            print(qa_info["answer"])

        
if __name__ == "__main__":
    # gen_civic_query()

    # handle_st_et()

    gen_civic_query_2()