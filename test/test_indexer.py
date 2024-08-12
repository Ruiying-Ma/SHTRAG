from enum import Flag
import json
import os
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG/")
from sht import SHTIndexerConfig, SHTIndexer
import numpy as np

def index_for_one_dataset_sht(dataset, use_hierarchy: bool, embedding_model: str):
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        queries = json.load(file)
    tree_suffix = ""
    if embedding_model != "sbert":
        tree_suffix += "-" + embedding_model
    index_info = []
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht{tree_suffix}"
    print(tree_dir)

    indexer_config = SHTIndexerConfig(use_hierarchy=use_hierarchy, distance_metric="cosine", query_embedding_model_name=embedding_model)
    indexer = SHTIndexer(indexer_config)

    for q_info in queries:
        
        json_name = q_info["file_name"] + ".json"

        with open(os.path.join(tree_dir, json_name), 'r') as file:
            sht = json.load(file)

        print(json_name)

        index_info.append({
            "id": q_info["id"],
            "query": q_info["query"],
            "indexes": indexer.index(query=q_info["query"], nodes=sht["nodes"]),
        })
        
        
    suffix = ""
    if use_hierarchy:
        suffix = "-r"
    assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht{suffix}{tree_suffix}/{dataset}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht{suffix}{tree_suffix}/{dataset}.json", 'w') as file:
        json.dump(index_info, file, indent=4)


if __name__ == "__main__":
    index_for_one_dataset_sht("civic", True, "te3small")
    index_for_one_dataset_sht("contract", True, "te3small")
    index_for_one_dataset_sht("qasper", True, "te3small")
    index_for_one_dataset_sht("civic", False, "te3small")
    index_for_one_dataset_sht("contract", False, "te3small")
    index_for_one_dataset_sht("qasper", False, "te3small")
    # index_for_one_dataset_sht("civic2", True, "te3small")
    # index_for_one_dataset_sht("civic2", False, "te3small")

    