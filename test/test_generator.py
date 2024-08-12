import json
import os
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG/")
from sht import SHTGeneratorConfig, SHTGenerator
import numpy as np

def generate_for_one_dataset_sht(dataset, indexer_use_hierarchy: bool, generator_use_hierarchy: bool, context_len: int, generator_use_raw_chunks: bool, indexer_model: str):
    indexer_suffix = ""
    if indexer_use_hierarchy:
        indexer_suffix = "-r"
    indexer_suffix += "-" + indexer_model
    print(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht{indexer_suffix}/{dataset}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/indexes/sht{indexer_suffix}/{dataset}.json", 'r') as file:
        indexes = json.load(file)

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/queries/{dataset}.json", 'r') as file:
        queries = json.load(file)

    assert len(indexes) == len(queries)

    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"

    generator_config = SHTGeneratorConfig(use_hierarchy=generator_use_hierarchy, use_raw_chunks=generator_use_raw_chunks, context_len=context_len)
    generator = SHTGenerator(generator_config)

    context_info = []

    for q_info, i_info in zip(queries, indexes):

        assert q_info['id'] == i_info["id"]
        assert q_info["query"] == i_info["query"]
        
        json_name = q_info["file_name"] + ".json"

        with open(os.path.join(tree_dir, json_name), 'r') as file:
            sht = json.load(file)

        print(json_name)

        context = generator.generate(candid_indexes=i_info["indexes"], nodes=sht["nodes"])

        context_info.append({
            "id": q_info["id"],
            "prompt_template": q_info["prompt_template"],
            "context": context,
        })
        
        
    generator_suffix = ""
    if generator_use_hierarchy:
        generator_suffix = "-g"
    if not generator_use_raw_chunks:
        generator_suffix += "-abs"
    assert not os.path.exists(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/sht{indexer_suffix+generator_suffix}/{dataset}-{context_len}.json")
    with open(f"/mnt/f/Research-2024-spring/SHTRAG/contexts/sht{indexer_suffix+generator_suffix}/{dataset}-{context_len}.json", 'w') as file:
        json.dump(context_info, file, indent=4)


if __name__ == "__main__":
    # generate_for_one_dataset_sht(dataset="civic2", indexer_use_hierarchy=True, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=True, indexer_model="te3small")
    # generate_for_one_dataset_sht(dataset="civic2", indexer_use_hierarchy=False, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=True, indexer_model="te3small")
    # generate_for_one_dataset_sht(dataset="civic2", indexer_use_hierarchy=True, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=True, indexer_model="te3small")
    # generate_for_one_dataset_sht(dataset="civic2", indexer_use_hierarchy=False, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=True, indexer_model='te3small')

    indexer_model = "te3small"
    indexer_hierarchies = [True, False]
    generator_hierarchies = [True, False]
    generator_raw = [True, False]
    datasets = ["civic", "contract", "qasper"]

    for generator_use_raw_chunks in generator_raw:
        for indexer_use_hierarchy in indexer_hierarchies:
            for generator_use_hierarchy in generator_hierarchies:
                for dataset in datasets:
                    generate_for_one_dataset_sht(dataset=dataset, indexer_use_hierarchy=indexer_use_hierarchy, generator_use_hierarchy=generator_use_hierarchy, context_len=1000, generator_use_raw_chunks=generator_use_raw_chunks, indexer_model=indexer_model)


    # generate_for_one_dataset_sht(dataset="civic", indexer_use_hierarchy=True, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="civic", indexer_use_hierarchy=False, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="civic", indexer_use_hierarchy=True, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="civic", indexer_use_hierarchy=False, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)

    # generate_for_one_dataset_sht(dataset="contract", indexer_use_hierarchy=True, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="contract", indexer_use_hierarchy=False, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="contract", indexer_use_hierarchy=True, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="contract", indexer_use_hierarchy=False, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)

    # generate_for_one_dataset_sht(dataset="qasper", indexer_use_hierarchy=True, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="qasper", indexer_use_hierarchy=False, generator_use_hierarchy=True, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="qasper", indexer_use_hierarchy=True, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)
    # generate_for_one_dataset_sht(dataset="qasper", indexer_use_hierarchy=False, generator_use_hierarchy=False, context_len=1000, generator_use_raw_chunks=False)