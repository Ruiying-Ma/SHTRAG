import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG/")
import json
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import os
from scipy import spatial
import numpy as np

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


def run_sbert_indexer_for_one_dataset(dataset):
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

        leaf_nodes = sorted([l for l in leaf_nodes_dict.values()], key=lambda leaf: leaf.index) # List[Node]

        assert list(range(len(leaf_nodes))) == [l.index for l in leaf_nodes]

        assert all([list(l.embeddings.keys()) == ["EMB"] for l in leaf_nodes])

        embeddings = [l.embeddings["EMB"] for l in leaf_nodes]

        query_embedding = custom_embedder.create_embedding(q_info["query"])

        distance_metric = spatial.distance.cosine

        distances = [distance_metric(query_embedding, embedding) for embedding in embeddings]

        indexes = np.argsort(distances)

        assert len(indexes) == len(leaf_nodes)
        assert set(indexes) == set(range(len(leaf_nodes)))


        index_info.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": [int(i) for i in indexes],
        })

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sbert/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)




# if __name__ == "__main__":
    # run_sbert_indexer_for_one_dataset("civic")