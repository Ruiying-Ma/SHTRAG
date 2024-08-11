
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
import json
import os
from scipy import spatial
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
import numpy as np
from sht import TextEmbedding3SmallModel

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

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name=""):
        pass

    def create_embedding(self, text):
        raise ValueError("should not achieve this part")


def te3small_add_embedding_for_one_dataset(dataset):
    assert dataset != "finance"
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"

    custom_summarizer = CustomSummarizationModel()
    custom_embedder = CustomEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    te3small_embedder = TextEmbedding3SmallModel(openai_key_path="/home/ruiying/Documents/Codebase/config/openai/config_openai.txt")

    files = os.listdir(tree_dir)
    for pickle_name in files:
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))

        leaf_nodes_dict = RA.tree.leaf_nodes # Dict[int, Node]

        leaf_nodes = sorted([l for l in leaf_nodes_dict.values()], key=lambda leaf: leaf.index) # List[Node]

        assert list(range(len(leaf_nodes))) == [l.index for l in leaf_nodes]

        te3small_nodes = []
        for leaf in leaf_nodes:
            embedding_info = te3small_embedder.create_embedding(text=leaf.text)
            assert leaf.index == len(te3small_nodes)
            te3small_nodes.append({
                "id": leaf.index,
                "embedding": embedding_info["embedding"],
            })
        json_name = pickle_name.replace(".pkl", ".json")
        assert not os.path.exists(os.path.join(f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/te3small", json_name))
        with open(os.path.join(f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/te3small", json_name), 'w') as file:
            json.dump(te3small_nodes, file, indent=4)

def te3small_indexing_for_one_dataset(dataset):
    assert dataset != "finance"
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/te3small"

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        data = json.load(file)

    te3small_embedder = TextEmbedding3SmallModel(openai_key_path="/home/ruiying/Documents/Codebase/config/openai/config_openai.txt")
    
    index_info = []
    for q_info in data:
        json_name = q_info["file_name"] + ".json"

        with open(os.path.join(tree_dir, json_name), 'r') as file:
            leaf_nodes_list = json.load(file)

        assert list(range(len(leaf_nodes_list))) == [l["id"] for l in leaf_nodes_list]

        # assert all(len(l["embedding"]) == 1 for l in leaf_nodes_list)
        embeddings = [l["embedding"] for l in leaf_nodes_list]

        query_embedding = te3small_embedder.create_embedding(text=q_info["query"])["embedding"]
        # assert len(query_embedding.shape) == 1
        # assert raw_query_embedding.shape == (1, query_embedding.shape[0])

        distance_metric = spatial.distance.cosine

        distances = [distance_metric(query_embedding, embedding) for embedding in embeddings]

        indexes = np.argsort(distances)

        assert len(indexes) == len(leaf_nodes_list)
        assert set(indexes) == set(range(len(leaf_nodes_list)))

        index_info.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": [int(i) for i in indexes],
        })
    assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/te3small/{dataset}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/te3small/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)

if __name__ == "__main__":
    # te3small_add_embedding_for_one_dataset("contract")
    # te3small_add_embedding_for_one_dataset("qasper")
    te3small_indexing_for_one_dataset("civic")
    te3small_indexing_for_one_dataset("contract")
    te3small_indexing_for_one_dataset("qasper")