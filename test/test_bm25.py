import sys
from turtle import heading
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG/")
import json
from typing import List
import tiktoken
from rank_bm25 import BM25Okapi
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import os
from sht import get_nondummy_ancestors


def bm25_indexing(chunks: List[str], query: str) -> List[int]:
    '''
    Return a list of the indexes sorted by the decreasing order of bm25 score. The indexes are 0-based. The 0th chunk has index 0. The tokenizer is "cl100k_base".

    Args:
        - chunks (List[str])

        - query (str)

    Return:
        - sorted list of the indexes
    '''
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenized_chunks = [tokenizer.encode(c) for c in chunks]
    tokenized_query = tokenizer.encode(query)

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)
    assert len(scores) == len(chunks)

    sorted_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return sorted_indices

class CustomQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        raise ValueError("should not achieve this part")

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):
        raise ValueError("should not achieve this part")

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

def test_bm25():
    chunks = ["Hello there good man!","It is quite windy in London","How is the weather today?", "windy", "Windy!!!! and sunny"]
    query = "windy London"

    sorted_indices = bm25_indexing(chunks, query)
    for id in sorted_indices:
        print(chunks[id])

def run_bm25_indexer_for_one_dataset(dataset):
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

        leaf_nodes_dict = RA.tree.leaf_nodes # Dict[int, Node]

        leaf_nodes = sorted([l for l in leaf_nodes_dict.values()], key=lambda leaf: leaf.index)

        assert list(range(len(leaf_nodes))) == [l.index for l in leaf_nodes]

        chunks = [l.text for l in leaf_nodes] 

        indexes = bm25_indexing(chunks, q_info["query"])

        assert len(indexes) == len(leaf_nodes)
        assert set(indexes) == set(range(len(leaf_nodes)))


        index_info.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": indexes,
        })

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/bm25/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)


def bm25_indexing_for_sht_and_shtr(dataset):
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        queries = json.load(file)

    index_info_no_hierarchy = []
    index_info_with_hierarchy = []
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"

    for q_info in queries:
        json_name = q_info["file_name"] + ".json"
        with open(os.path.join(tree_dir, json_name), 'r') as file:
            sht = json.load(file)

        print(json_name)

        query = q_info["query"]

        chunks_no_hierarchy = []
        chunks_with_hierarchy = []

        m_id_to_pos = dict()

        for node in sht["nodes"]:
            if node["is_dummy"]:
                continue
            node_id = node["id"]
            ancestors = get_nondummy_ancestors(sht["nodes"], node_id)
            if len(ancestors) > 0:
                assert sorted(ancestors) == ancestors
                assert all([(aid >= 0 and aid < len(sht["nodes"]) for aid in ancestors)])
                assert all([not sht["nodes"][aid]["is_dummy"] for aid in ancestors])
                assert all([sht["nodes"][aid]["type"] != "text" for aid in ancestors])
                assert all([sht["nodes"][aid]["heading"] != "" for aid in ancestors])
                assert sorted(ancestors) == ancestors
            ancestor_string = "\n\n".join([sht["nodes"][aid]["heading"] for aid in ancestors])
            if ancestor_string != "":
                ancestor_string += "\n\n"
            
            heading_string = node["heading"]
            if heading_string != "":
                heading_string += "\n\n"

            if node["type"] == "text":
                assert set(node["embeddings"].keys()) == set(["texts", "hybrid"])
                assert node["heading"] == "" and heading_string == ""
            else:
                assert node["type"] in ["head", "list"]
                assert set(node["embeddings"].keys()) == set(["texts", "hybrid", "heading"])
                assert node["heading"] != ""

            for chunk_id, text in enumerate(node["texts"]):
                assert len(chunks_no_hierarchy) == len(chunks_with_hierarchy)
                id = len(chunks_no_hierarchy)
                m_id_to_pos[id] = {
                    "node_id": node["id"],
                    "chunk_id": chunk_id,
                }
                chunks_no_hierarchy.append(heading_string + text)
                chunks_with_hierarchy.append(ancestor_string + heading_string + text)

        assert len(chunks_no_hierarchy) == len(chunks_with_hierarchy)

        sorted_indexes_no_hierarchy = bm25_indexing(chunks_no_hierarchy, query)
        index_info_no_hierarchy.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": [m_id_to_pos[sinh] for sinh in sorted_indexes_no_hierarchy]
        })
        

        sorted_indexes_with_hierarchy = bm25_indexing(chunks_with_hierarchy, query)
        index_info_with_hierarchy.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": [m_id_to_pos[siwh] for siwh in sorted_indexes_with_hierarchy]
        })

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht-bm25/{dataset}.json", 'w') as file:
        json.dump(index_info_no_hierarchy, file, indent=4)
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht-r-bm25/{dataset}.json", 'w') as file:
        json.dump(index_info_with_hierarchy, file, indent=4)


if __name__ == "__main__":
    bm25_indexing_for_sht_and_shtr("civic")
    bm25_indexing_for_sht_and_shtr("contract")
    bm25_indexing_for_sht_and_shtr("qasper")