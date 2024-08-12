import logging
from typing import Dict, List
from scipy import spatial
import numpy as np

from .EmbeddingModels import DPRQueryEmbeddingModel, SBertEmbeddingModel, TextEmbedding3SmallModel


class SHTIndexerConfig:
    def __init__(self, use_hierarchy: bool, distance_metric: str, query_embedding_model_name: str, openai_key_path: str="/home/ruiying/Documents/Codebase/config/openai/config_openai.txt"):
        '''
        Args:
            - use_hierarchy (bool): whether use hybrid- or texts- embeddings
            - distance_metrics (str): "cosine", "L1", "L2", "Linf"
        '''
        if not isinstance(use_hierarchy, bool):
            raise ValueError("use_hierarchy must be a boolean value")
        
        if not isinstance(distance_metric, str) or not distance_metric in ["cosine", "L1", "L2", "Linf"]:
            raise ValueError("distance_metric must be one of 'cosine', 'L1', 'L2', 'Linf'")

        if not isinstance(query_embedding_model_name, str) or not query_embedding_model_name in ["dpr", "sbert", "te3small"]:
            raise ValueError("query_embedding_model must be onde of 'sbert', 'dpr', or 'te3small'")

        self.use_hierarhy = use_hierarchy

        if distance_metric == "cosine":
            self.distance_metric = spatial.distance.cosine
        elif distance_metric == "L1":
            self.distance_metric = spatial.distance.cityblock
        elif distance_metric == "L2":
            self.distance_metric = spatial.distance.euclidean
        else:
            self.distance_metric = spatial.distance.chebyshev

        self.query_embedding_model_name = query_embedding_model_name
        self.openai_key_path = openai_key_path

            
class SHTIndexer:
    def __init__(self, config: SHTIndexerConfig):
        if not isinstance(config, SHTIndexerConfig):
            raise ValueError("config must be an instance of SHTIndexerConfig")

        self.use_hierarchy = config.use_hierarhy
        self.distance_metric = config.distance_metric
        self.query_embedder = None
        if config.query_embedding_model_name == "sbert":
            self.query_embedder = SBertEmbeddingModel()
        elif config.query_embedding_model_name == "dpr":
            self.query_embedder = DPRQueryEmbeddingModel()
        elif config.query_embedding_model_name == "te3small":
            self.query_embedder = TextEmbedding3SmallModel(openai_key_path=config.openai_key_path)
        else:
            raise ValueError(f"Unknown embedding model: {config.query_embedding_model_name}")


    def index(self, query: str, nodes: List[Dict]) -> List[Dict]:
        '''
        Return the sorted indexes of chunks. The position of a chunk is denoted by a dict:
        - "node_id" (int)
        - "chunk_id" (int)
        '''
        log_info = "hybrid" if self.use_hierarchy else "texts"
        logging.info(f"Indexing ({log_info}) for query '{query}'...")

        m_id_to_pos = dict() # pos is dictionary {"node_id":, "chunk_id":}
        chunk_embeddings = []
        for node in nodes:
            if node["is_dummy"]:
                continue
            if self.use_hierarchy:
                embeddings = [e for e in node["embeddings"]["hybrid"]]
            else:
                embeddings = [e for e in node["embeddings"]["texts"]]
            assert all([isinstance(e, list) for e in embeddings])
            
            for chunk_id, embedding in enumerate(embeddings):
                id = len(chunk_embeddings)
                assert id not in m_id_to_pos
                m_id_to_pos[id] = {
                    "node_id": node["id"],
                    "chunk_id": chunk_id
                }
                chunk_embeddings.append(embedding)
        
        query_embedding = self.query_embedder.create_embedding(query)["embedding"]
        assert isinstance(query_embedding, list)

        distances = [self.distance_metric(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]

        sorted_ids = np.argsort(distances)

        return [m_id_to_pos[i] for i in sorted_ids]





