import subprocess
import json
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
from sht import ClusteringOracleConfig, ClusteringOracle, SHTBuilderConfig, SHTBuilder, SHTIndexerConfig, SHTIndexer, SHTGenerator, SHTGeneratorConfig
import os

def vgt_for_heading_extraction(pdf_path):
    curl_command = f'''curl -X POST -F 'file=@{pdf_path}' localhost:5060'''
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

    out_path = pdf_path.replace(".pdf", ".dla.json")
    with open(out_path, 'w') as file:
        json.dump(json.loads((result.stdout)), file, indent=4)

def build_sht_skeleton(pdf_path, dla_json_path):
    '''
    Args:
    - out_path: the result of DLA (vgt_for_heading_extraction)
    '''
    
    with open(dla_json_path, 'r') as file:
        object_dicts_list = json.load(file)
    
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=dla_json_path.replace(".dla", ".cluster"), 
    )

    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    new_object_dicts_list = clustering_oracle.cluster(
        pdf_path=pdf_path,
        object_dicts_list=object_dicts_list,
    )

    sht_builder_config = SHTBuilderConfig(
        store_json=dla_json_path.replace(".dla", ".sht"), 
        load_json=None, 
        chunk_size=100, 
        summary_len=100,
        embedding_model_name="sbert",
    )
    sht_builder = SHTBuilder(sht_builder_config)
    sht_builder.build(new_object_dicts_list) # or, sht_builder.build(None) if SHT already existed
    sht_builder.store2json()
    sht_builder.visualize()

def load_sht_skeleton(sht_json_path):
    sht_builder_config = SHTBuilderConfig(
        store_json=sht_json_path.replace(".json", ".add-summaries.json"), 
        load_json=sht_json_path, 
        chunk_size=100, 
        summary_len=100,
        embedding_model_name="sbert",
    )
    sht_builder = SHTBuilder(sht_builder_config)
    sht_builder.build(None) # or, sht_builder.build(None) if SHT already existed
    sht_builder.check()
    sht_builder.store2json()
    sht_builder.visualize()

def add_summaries_to_sht(sht_skeleton_json_path):
    sht_builder_config = SHTBuilderConfig(
        store_json=sht_skeleton_json_path.replace(".json", ".sht.add-summaries.json"), 
        load_json=sht_skeleton_json_path, 
        chunk_size=100, 
        summary_len=100,
        embedding_model_name="sbert",
    )
    sht_builder = SHTBuilder(sht_builder_config)
    sht_builder.build(None) # or, sht_builder.build(None) if SHT already existed
    stats = sht_builder.add_summaries()
    sht_builder.check()
    sht_builder.store2json()
    sht_builder.visualize()

    with open(sht_skeleton_json_path.replace(".json", ".add-summaries.stats.json"), 'w') as file:
        json.dump(stats, file, indent=4)

def add_embeddings_to_sht(sht_json_path, embedding):
    sht_builder_config = SHTBuilderConfig(
        store_json=sht_json_path.replace(".json", ".sht.add-embeddings.json"), 
        load_json=sht_json_path, 
        chunk_size=100, 
        summary_len=100,
        embedding_model_name=embedding,
    )
    sht_builder = SHTBuilder(sht_builder_config)
    sht_builder.build(None) # or, sht_builder.build(None) if SHT already existed
    node_ids_list = list(range(len(sht_builder.tree["nodes"])))
    stats = sht_builder.add_embeddings(node_ids_list)
    sht_builder.check()
    sht_builder.store2json()
    sht_builder.visualize()

    with open(sht_json_path.replace(".json", ".add-embeddings.stats.json"), 'w') as file:
        json.dump(stats, file, indent=4)

def index(sht_full_json_path, query):
    indexer_config = SHTIndexerConfig(
        use_hierarchy=True, 
        distance_metric="cosine", 
        query_embedding_model_name="sbert"
    )
    indexer = SHTIndexer(indexer_config)

    with open(sht_full_json_path, 'r') as file:
        sht = json.load(file)

    index_info = {
        "query": query,
        "indexes": indexer.index(query=query, nodes=sht["nodes"])
    }

    with open(sht_full_json_path.replace(".json", ".index.hierarchy.json"), 'w') as file:
        json.dump(index_info, file, indent=4)

def generate(sht_full_json_path, index_json_path, context_len, use_hierarchy):
    generator_config = SHTGeneratorConfig(use_hierarchy=use_hierarchy, use_raw_chunks=True, context_len=context_len)
    generator = SHTGenerator(generator_config)

    with open(index_json_path, 'r') as file:
        i_info = json.load(file)
    with open(sht_full_json_path, 'r') as file:
        sht = json.load(file)

    context = generator.generate(
        candid_indexes=i_info["indexes"],
        nodes=sht["nodes"]
    )

    with open(sht_full_json_path.replace(".json", f".context.{context_len}.{use_hierarchy}.txt"), 'w') as file:
        file.write(context)


    

# if __name__ == "__main__":
#     pdf_path = "/mnt/f/Research-2024-spring/SHTRAG/example/1701.04056v1.pdf"
#     # vgt_for_heading_extraction(pdf_path)
#     dla_json_path = "/mnt/f/Research-2024-spring/SHTRAG/example/1701.04056v1.dla.json"
#     # build_sht_skeleton(pdf_path, dla_json_path)
#     # sht_json_path = "/mnt/f/Research-2024-spring/SHTRAG/example/1701.04056v1.c-correct-sht.json"
#     # load_sht_skeleton(sht_json_path)
#     # add_summaries_to_sht(sht_json_path)
#     # add_embeddings_to_sht(sht_json_path, "sbert")
#     sht_full_json_path = "/mnt/f/Research-2024-spring/SHTRAG/example/1701.04056v1.c-correct-sht.sht.add-embeddings.json"
#     query = '''What section discusses the RNN model where the last hidden state of the previous utterance serves as the initial RNN state of the next?'''
#     # index(sht_full_json_path, query)

#     index_json_path = "/mnt/f/Research-2024-spring/SHTRAG/example/1701.04056v1.c-correct-sht.sht.add-embeddings.index.hierarchy.json"

#     generate(sht_full_json_path,index_json_path, context_len=200, use_hierarchy=False)