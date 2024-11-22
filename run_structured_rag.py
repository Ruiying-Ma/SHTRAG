from structured_rag import StructuredRAG
import argparse
import os
import json

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="root dir (pdf file is under root_dir/pdf/)")
    parser.add_argument("--chunk-size", type=int, required=False, default=100, help="size of a chunk (i.e., the length of the context of a newly added leaf)")
    parser.add_argument("--summary-len", type=int, required=False, default=100, help="length of a recursively generated summary (i.e., the length of the context of an original SHT node)")
    parser.add_argument("--node-embedding-model", type=str, required=False, choices=["sbert", "dpr", "te3small"], default="sbert", help="the embedding model for SHT nodes")
    parser.add_argument("--query-embedding-model", type=str, required=False, choices=["sbert", "dpr", "te3small"], default="sbert", help="the embedding model for the query")
    parser.add_argument("--summarization-model", type=str, required=False, choices=["gpt-4o-mini", "empty"], default="gpt-4o-mini", help="the summarization model")
    parser.add_argument("--embed-hierarchy", type=str_to_bool, required=False, default=True, help="whether to embed the hierarchical information")
    parser.add_argument("--distance-metric", type=str, choices=["cosine", "L1", "L2", "Linf"], required=False, default="cosine", help="the distance metric in the embedding space")
    parser.add_argument("--context-hierarchy", type=str_to_bool, required=False, default=True, help="whether to recover hierarchical information in the final context")
    parser.add_argument("--context-raw", type=str_to_bool, required=False, default=True, help="whether to retrieve the newly added leaves (i.e., the chunks of the document) for the final context")
    parser.add_argument("--context-len", type=int, required=False, default=1000, help='the length of the final context')
    
    args = parser.parse_args()

    print(
        args.root_dir,
        args.chunk_size, 
        args.summary_len, 
        args.node_embedding_model, 
        args.query_embedding_model, 
        args.summarization_model,
        args.embed_hierarchy,
        args.distance_metric,
        args.context_hierarchy,
        args.context_raw,
        args.context_len
    )

    queries_path = os.path.join(args.root_dir, "queries.json")
    with open(queries_path, 'r') as file:
        queries_info = json.load(file)
    
    for qid, query_info in enumerate(queries_info):
        assert qid == query_info["id"]
        rag = StructuredRAG(
            root_dir=args.root_dir,
            chunk_size=args.chunk_size,
            summary_len=args.summary_len,
            node_embedding_model=args.node_embedding_model,
            query_embedding_model=args.query_embedding_model,
            summarization_model=args.summarization_model,
            embed_hierarchy=args.embed_hierarchy,
            distance_metric=args.distance_metric,
            context_hierarchy=args.context_hierarchy,
            context_raw=args.context_raw,
            context_len=args.context_len
        )
        rag.generate_context(
            name=query_info["file_name"],
            query=query_info["query"],
            query_id=qid,
        )