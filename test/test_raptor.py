import json
import os
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
from typing import Dict 
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging
import time
import tiktoken
from sentence_transformers import SentenceTransformer
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class CustomQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        raise ValueError("should not achieve this part")
        answer = "Your answer here"
        return answer

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):
        raise ValueError("should not achieve this part")

class GPT4oMiniSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.user_prompt_template = "Please summarize the following text in no more than four sentences. Ensure that the summary includes all key details.\n\n[Start of the text]\n{text}\n[End of the text]"
        self.stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "time": 0.0
        }
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):

        try:
            logging.info(f"{self.model} summarizing (in {max_tokens} tokens) text: {context[0:min(10, len(context))]}...")
            
            client = OpenAI()

            user_prompt = self.user_prompt_template.format(text=context)

            start_time = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=max_tokens,
            )
            end_time = time.time()

            summary = response.choices[0].message.content
            self.stats["input_tokens"] += len(self.tokenizer.encode(user_prompt))
            self.stats["output_tokens"] += len(self.tokenizer.encode(summary))
            self.stats["time"] += end_time - start_time

            # summary = ""
            # self.stats["input_tokens"] += 1
            # self.stats["output_tokens"] += 1
            # self.stats["time"] += 1.0

            return summary

        except Exception as e:
            print(e)
            return e

class MySBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)
        self.time = 0.0

    def create_embedding(self, text):
        start = time.time()
        embedding = self.model.encode(text)
        end = time.time()
        self.time += (end - start)
        return embedding

def run_raptor_for_one_doc_text(doc_text: str, save_path: str) -> Dict:
    '''
    Store raptor tree in both .pkl and .json form.

    Args:
        - doc_text (str)
        - save_path (str): the path to save the raptor tree (.pkl)

    Return:
        - stats (Dict):
            - input_tokens (int)
            - output_tokens (int)
            - summarization_time (float)
            - embedding_time (float)
    '''
    if isinstance(save_path, str):
        if not save_path.endswith(".pkl"):
            raise ValueError("save_path must be a path to a pickle file that ends with .pkl")
        elif not os.path.exists(os.path.dirname(save_path)):
            raise ValueError("save_path directory doesn't exist")
    else:
        raise ValueError("save_path must be a valid string: the built tree should be stored in files")
    
    openai_key_path = "/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"
    with open(openai_key_path, 'r') as file:
        os.environ["OPENAI_API_KEY"] = file.read().replace("\n", "").strip()

    custom_summarizer = GPT4oMiniSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    RA = RetrievalAugmentation(config=custom_RAConfig)
    RA.add_documents(doc_text)
    RA.save(save_path)

    summary_stats = custom_summarizer.stats
    embedding_time = custom_embedder.time
    # assert summary_stats["input_tokens"] == len(RA.tree.all_nodes) - len(RA.tree.leaf_nodes)

    return {
        "input_tokens": summary_stats["input_tokens"],
        "output_tokens": summary_stats["output_tokens"],
        "summarization_time": summary_stats["time"],
        "embedding_time": embedding_time,
    }

def run_raptor_for_one_dataset(dataset, start, end):
    in_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"
    out_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"
    files = os.listdir(in_dir)
    json_files = [f for f in files if f.endswith(".json")]
    assert end <= len(json_files)
    tot_stats = {
        "tot_input_tokens": 0,
        "tot_output_tokens": 0,
        "tot_summarization_time": 0.0,
        "tot_embedding_time": 0.0,
        "details": []
    }
    for json_name in json_files[start:end]:
        print("=========================RAPTOR================================")
        print(f"Creating RA for {json_name}...")
        assert json_name.endswith(".json")
        with open(os.path.join(in_dir, json_name), 'r') as file:
            data = json.load(file)

        doc_text = data["full_text"]

        save_file_name = json_name.replace(".json", ".pkl")
        stats = run_raptor_for_one_doc_text(doc_text, os.path.join(out_dir, save_file_name))

        tot_stats["tot_input_tokens"] += stats["input_tokens"]
        tot_stats["tot_output_tokens"] += stats["output_tokens"]
        tot_stats["tot_summarization_time"] += stats["summarization_time"]
        tot_stats["tot_embedding_time"] += stats["embedding_time"]
        tot_stats["details"].append(stats)

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/{dataset}[{start}:{end}]_raptor_build_and_embedding_stats.json", 'w') as file:
        json.dump(tot_stats, file, indent=4)

def run_raptor_indexer_for_one_dataset(dataset):
    assert dataset != "finance"
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        data = json.load(file)

    index_info = []

    custom_summarizer = CustomSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    for q_info in data:
        pickle_name = q_info["file_name"] + ".pkl"
        
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))

        top_k = len(RA.tree.all_nodes)

        context, layer_info = RA.retrieve(
            question=q_info["query"],
            top_k=top_k,
            max_tokens=sys.maxsize,
            collapse_tree=True,
            return_layer_information=True,
        )

        assert len(layer_info) == len(RA.tree.all_nodes)

        index_info.append(
            {
                "id": q_info["id"],
                "query": q_info["query"],
                "indexes": [info["node_index"] for info in layer_info],
            }
        )

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/raptor/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)
        

def generate_context(dataset, config, context_len):
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/{config}/{dataset}.json", 'r') as file:
        indexes = json.load(file)
    
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        queries = json.load(file)

    assert len(indexes) == len(queries)

    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"

    context_info = []

    custom_summarizer = CustomSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    for q_info, i_info in zip(queries, indexes):
        assert q_info['id'] == i_info["id"]
        assert q_info["query"] == i_info["query"]

        pickle_name = q_info["file_name"] + ".pkl"
        
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))

        context = ""
        context_token_count = 0
        for node_index in i_info["indexes"]:
            text = RA.tree.all_nodes[int(node_index)].text
            assert text != ""
            text += "\n\n"
            text_token_count = len(tokenizer.encode(text))
            if context_token_count + text_token_count <= context_len:
                context += text
                context_token_count += text_token_count
            else:
                break
        
        assert len(tokenizer.encode(context)) == context_token_count
        assert context_token_count <= context_len

        context_info.append({
            "id": q_info["id"],
            "prompt_template": q_info["prompt_template"],
            "context": context,
        })
    assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/{dataset}-{context_len}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}/{dataset}-{context_len}.json", 'w') as file:
        json.dump(context_info, file, indent=4)


def generate_context_sorted(dataset, config, context_len):
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/{config}/{dataset}.json", 'r') as file:
        indexes = json.load(file)
    
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        queries = json.load(file)

    assert len(indexes) == len(queries)

    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"

    context_info = []

    custom_summarizer = CustomSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    for q_info, i_info in zip(queries, indexes):
        assert q_info['id'] == i_info["id"]
        assert q_info["query"] == i_info["query"]

        pickle_name = q_info["file_name"] + ".pkl"
        
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))

        context_node_id = []
        context_token_count = 0
        for node_index in i_info["indexes"]:
            text = RA.tree.all_nodes[int(node_index)].text
            assert text != ""
            text += "\n\n"
            text_token_count = len(tokenizer.encode(text))
            if context_token_count + text_token_count <= context_len:
                context_node_id.append(node_index)
                context_token_count += text_token_count
            else:
                break
        
        sorted_context_node_id = sorted(context_node_id)
        assert sorted(sorted_context_node_id) == sorted_context_node_id
        context = ""
        for node_index in sorted_context_node_id:
            text = RA.tree.all_nodes[int(node_index)].text
            assert text != ""
            text += "\n\n"
            context += text

        assert len(tokenizer.encode(context)) == context_token_count
        assert context_token_count <= context_len

        context_info.append({
            "id": q_info["id"],
            "prompt_template": q_info["prompt_template"],
            "context": context,
        })
    assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}-o/{dataset}-{context_len}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/{config}-o/{dataset}-{context_len}.json", 'w') as file:
        json.dump(context_info, file, indent=4)


def raptor_num_nodes(dataset):
    import numpy as np
    custom_summarizer = CustomSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"
    files = os.listdir(tree_dir)
    node_count = []
    for pickle_name in files:
        assert pickle_name.endswith(".pkl")
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))
        node_count.append(len(RA.tree.all_nodes))
    assert len(files) == len(node_count)
    print(np.mean(node_count))


if __name__ == "__main__":
    generate_context_sorted(dataset="civic", config="te3small", context_len=1000)
    generate_context_sorted(dataset="contract", config="te3small", context_len=1000)
    generate_context_sorted(dataset="qasper", config="te3small", context_len=1000)
