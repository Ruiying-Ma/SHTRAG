
'''
https://bekushal.medium.com/llms-for-question-answering-and-dense-passage-retrieval-dpr-209661bb7518

... one should use the DPRQuestionEncoder to encode the user query, and DPRContextEncoder to encode the passages, and then compute cosine similarities...

'''
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
import json
import os
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from scipy import spatial
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
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

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name=""):
        pass

    def create_embedding(self, text):
        raise ValueError("should not achieve this part")


def dpr_add_embedding_for_one_dataset(dataset):
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

    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    dpr_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    files = os.listdir(tree_dir)
    for pickle_name in files:
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))

        leaf_nodes_dict = RA.tree.leaf_nodes # Dict[int, Node]

        leaf_nodes = sorted([l for l in leaf_nodes_dict.values()], key=lambda leaf: leaf.index) # List[Node]

        assert list(range(len(leaf_nodes))) == [l.index for l in leaf_nodes]

        dpr_nodes = []
        for leaf in leaf_nodes:
            # print(leaf.text, len(tiktoken.get_encoding("cl100k_base").encode(leaf.text)))##################
            input_ids = dpr_tokenizer(leaf.text, return_tensors="pt", max_length=512)["input_ids"]
            embedding = dpr_model(input_ids).pooler_output
            assert leaf.index == len(dpr_nodes)
            dpr_nodes.append({
                "id": leaf.index,
                "embedding": embedding.detach().numpy().tolist(),
            })
        json_name = pickle_name.replace(".pkl", ".json")
        # assert not os.path.exists(os.path.join(f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/dpr", json_name))
        with open(os.path.join(f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/dpr", json_name), 'w') as file:
            json.dump(dpr_nodes, file, indent=4)

def dpr_indexing_for_one_dataset(dataset):
    assert dataset != "finance"
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/dpr"

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        data = json.load(file)

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    
    index_info = []
    for q_info in data:
        json_name = q_info["file_name"] + ".json"

        with open(os.path.join(tree_dir, json_name), 'r') as file:
            leaf_nodes_list = json.load(file)

        assert list(range(len(leaf_nodes_list))) == [l["id"] for l in leaf_nodes_list]

        assert all(len(l["embedding"]) == 1 for l in leaf_nodes_list)
        embeddings = [np.array(l["embedding"][0]) for l in leaf_nodes_list]

        input_ids = tokenizer(q_info["query"], return_tensors="pt")["input_ids"]
        raw_query_embedding = model(input_ids).pooler_output.detach().numpy()
        query_embedding = raw_query_embedding[0]
        assert len(query_embedding.shape) == 1
        assert raw_query_embedding.shape == (1, query_embedding.shape[0])

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

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/dpr/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)


def sht_dpr_add_embedding_for_one_dataset(dataset):
    assert dataset != "finance"
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"

    custom_summarizer = CustomSummarizationModel()
    custom_embedder = CustomEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    dpr_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

if __name__ == "__main__":
    # dpr_add_embedding_for_one_dataset("contract")
    # dpr_add_embedding_for_one_dataset("qasper")
    dpr_indexing_for_one_dataset("civic")
    dpr_indexing_for_one_dataset("contract")
    dpr_indexing_for_one_dataset("qasper")