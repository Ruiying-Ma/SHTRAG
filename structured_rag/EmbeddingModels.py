import logging
from abc import ABC, abstractmethod
from typing import Dict
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        '''
        Embed the text. 

        Return: stats info
            - stats (Dict):
                - embedding (list)
                - time (float)
        '''
        pass

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text) -> Dict:
        logging.info(f"SBert is creating embedding for text: {text[:min(len(text), 10)]}...")
        start = time.time()
        embedding =  self.model.encode(text).tolist()
        end = time.time()
        return {
            "embedding": embedding,
            "time": end - start,
        }

class DPRContextEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/dpr-ctx_encoder-multiset-base"):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRContextEncoder.from_pretrained(model_name)

    def create_embedding(self, text) -> Dict:
        logging.info(f"DPRContextEncoder is creating embedding for text: {text[:min(len(text), 10)]}...")
        start = time.time()
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=512)["input_ids"]
        embedding = self.model(input_ids).pooler_output.detach().numpy().flatten().tolist()
        end = time.time()
        return {
            "embedding": embedding,
            "time": end - start,
        }

class DPRQueryEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/dpr-question_encoder-multiset-base"):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)

    def create_embedding(self, text) -> Dict:
        logging.info(f"DPRQueryEncoder is creating embedding for text: {text[:min(len(text), 10)]}...")
        start = time.time()
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=512)["input_ids"]
        embedding = self.model(input_ids).pooler_output.detach().numpy().flatten().tolist()
        end = time.time()
        return {
            "embedding": embedding,
            "time": end - start,
        }

class TextEmbedding3SmallModel(BaseEmbeddingModel):
    def __init__(self, openai_key_path: str, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        # with open(openai_key_path, 'r') as file:
        #     self.client = OpenAI(api_key=file.read().replace("\n", "").strip())
        load_dotenv(openai_key_path)
        self.client = OpenAI(api_key=os.getenv("API_KEY"))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text) -> Dict:
        assert self.client is not None
        logging.info(f"text-embedding-3-small is creating embedding for text: {text[:min(len(text), 10)]}...")
        start = time.time()
        embedding = self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding
        end = time.time()
        return {
            "embedding": embedding,
            "time": end - start,
        }

