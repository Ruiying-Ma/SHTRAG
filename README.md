# Dependencies
`SHTRAG/raptor/requirements.txt`
# How to run SHTRAG on **one document** and **one query**
Recommed: first check `SHTRAG/sht/README.md`

1. Use [VGT](https://github.com/huridocs/pdf-document-layout-analysis) (a DLA model) to classify document objects
    ```python 
    pdf_path = "absolute-path-to-pdf"
    curl_command = f'''curl -X POST -F 'file=@{pdf_path}' localhost:5060'''
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

    out_path = "absolute-path-to-destination-json-file"
    with open(out_path, 'w') as file:
        json.dump(json.loads((result.stdout)), file, indent=4)
    ```
2. Build SHT
    ```python
    with open(out_path, 'r') as file:
        object_dicts_list = json.load(file)
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=None, 
        object_dicts_list=object_dicts_list,
    )
    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    new_object_dict_lists = clustering_oracle.cluster(pdf_path)

    sht_builder_config = SHTBuilderConfig(
        store_json=os.path.join(out_dir, json_name), 
        load_json=None, 
        chunk_size=100, 
        summary_len=100,
        embedding_model_name=embedding_mode_name,
        openai_key_path=path, 
    )
    sht_builder = SHTBuilder(sht_builder_config)
    sht_builder.build(new_object_dicts_list) # or, sht_builder.build(None) if SHT already existed
    summary_stats = sht_builder.add_summaries()
    embedding_stats = sht_builder.add_embeddings(node_ids_list)
    sht_builder.store2json()
    sht_builder.visualize()
    ```

3. Indexing
    ```python
    indexer_config = SHTIndexerConfig(
        use_hierarchy=use_hierarchy,
        distance_metric="cosine",
        query_embedding_model_name=embedding_model,
    )
    indexer = SHTIndexer(indexer_config)
    nodes = sht_builder.tree
    query = "your-query"
    indexes = indexer.index(query, nodes)
    ```

4. Generating Context
    ```python
    generator_config = SHTGeneratorConfig(
        use_hierarchy=generator_use_hierarchy,
        use_raw_chunks=generator_use_raw_chunks, 
        context_len=context_len
    ) 
    generator = SHTGenerator(generator_config)
    context = generator.generate(indexes, nodes)
    ```

5. Answer questions using a QA model

# Codes for Experiments
- `sht/`

    The codes for SHTRAG. Detailed description can be seen in `SHTRAG/sht/README.md`.

- `raptor/`

    The codes for RAPTOR. Clone from repo of the paper. Make one change to `split_text()` in `SHTRAG/raptor/utils.py`: change the way of creating chunks. Same chunking method as that of SHTRAG (in `SHTRAG/sht/utils.py`).

    Example usage of RAPTOR can be seen in `SHTRAG/test/test_raptor.py` step by step:
    1. Tree building: `run_raptor_for_one_doc_text()` runs raptor's on the texts of a document, and then save the tree (and its visualization) in a pickle (txt) file.
    2. Indexing: `run_raptor_indexer_for_one_dataset()` creates indexes for tree nodes, for each query.
    3. Context generation: 
        - `generate_context()` generates the final contexts given the token limit. Note that we don't limit the number of nodes. (i.e., not top_k). The nodes are ordered in priority order returned by indexing process.
        - `generate_context_sorted()` generates the contexts in reading order. 

- `dpr/`

    Codes for DPR. `DPRContextEncoder` embeds the texts to be retrieved. `DPRQuestionEncoder` embeds the query.

    In `SHTRAG/dpr/test_dpr.py`:
    - `dpr_add_embedding_for_one_dataset()`: naive chunking, embedding process
    - `dpr_indexing_for_one_dataset()`: naive chunking, indexing process
    - `sht_dpr_add_embedding_for_one_dataset()`: Build SHT using DPR for embedding

- `test/`

    - `civic_queries.py`: create civic queries
    - `contract_queries.py`: reformat ContractNLI queries
    - `qasper_queries.py`: reformat Qasper queries
    - `eval_context.py`: 
        - `{dataset}_wrong_answers(configs)`: extract the wrong answers for a dataset, given the set of experiment configs (e.g., sht, raptor, bm25, etc.)
        - `contract_answers_and_contexts(configs)`: create the confusion matrix of ContractNLI

    - `handle_qapser.py`: `build_sht()` build the SHT for Qasper using the intrinsic SHT provided by Qasper
    - `test_bm25.py`: 
        - `run_bm25_indexer_for_one_dataset()`: naive chunking and indexing using BM25
        - `bm25_indexing_for_sht_and_shtr()`: use BM25 on SHTRAG
    - `test_sbert.py`: naive chunkign and indexig use SBERT
    - `test_gold_context.py`: 
        - `create_contract_batch_gold_context()`: create the gold context for each query in ContractNLI (provided by ContractNLI spans)
        - `create_qasper_batch_gold_context_long_short()`: create the gold context for each query in Qasper. Long gold context refers to the evidence provided by Qasper. Short gold context refers to the highlighted_evidence provided by Qasper.
        - `create_contract_batch_full_contest()`: use full document as the context for ContractNLI.
    
    - `test.py`, `test_indexer.py`, `test_generator.py`: usage of SHTRAG. See `SHTRAG/sht/README.md`
    - `test_raptor.py`: See section `raptor/` in this README.md.

- `te3small/`

    Codes for text-embedding-3-small. 
    
    `test_te3small.py`: Naive chunking with this embedding model.

- `batches/`

    Codes for using batch api of OPENAI. Steps:
    1. Create a .jsonl file as a batch file, following a specific formatting rule
    2. Upload jsonl file to OPENAI
    3. Create a batch job on the uploaded batch file. Then the server begins to run the job.
    4. Check the status of a batch job.
    5. Retrieve the responses of the each query in the batch job after completion.
    6. Delete the batch file and the batch job.

    You can use CLI `curl https://api.openai.com/v1/files -H "Authorization: Bearer your-openai-key"` to check all the batch files on OPENAI server

    You can use CLI `curl https://api.openai.com/v1/batches -H "Authorization: Bearer your-openai-key"` to check all the batch jobs on OPENAI server

    - `create_batch_job.py`: 
        - `create_batch_sht_civic_civic2_contract_qasper_1000()`: an example usage of step 1
        - `upload_batch()`: usage of step 2. Upload the create jsonl file to the OPENAI server. A log file will be created on your local machine. Note that the size of the batch file should not exceed 100MB, and the number of requests in that file (i.e., the number of lines in the jsonl file) should not exceed 50,000.
        - `creaet_job()`: usage of step 3. Create a batch job on the uploaded batch file. The `batch_input_file_id` is stored in the log file `log_json_name` you've created when running `upload_batch()`. After the job is created, the log file will be updated with the creation action. You can find the batch_id in the updated log file.
    - `check_batch.py`
        - `check_batch()`: usage of step 4. Check the status of the batch job. The batch_id can be seen in the log file. You can see how many requests in the batch job have been completed. When the batch job is finished, its status should be `completed` (you can check it in the log file). Every time you execute `check_batch()`, the log file will be updated. If errors occur, you can also see this from the log file.
    - `retrieve_batch.py`
        - `reetrieve_response_batch()`: usage of step 5. You can do this only when the status of the batch job is `completed`. The responses from the server are downloead to you local machine as a jsonl file. 
    - `delete_batch.py`
        - `delete_file()`: usage of step 6. You can delete the batch job, batch file, and error file (if error occurs) after you retrieve the responses.

- `answers/`

    - `extract_answers.py`: extract answers from the responses retrieved from the OPENAI server (i.e., the jsonl file)

    - `eval_answers.py`: evaluate the answers of different datasets.

- `llm_judge/`

    LLM-as-a-judge for Qasper.

    - `create_tasks.py`: create jsonl batch file 
    - `extract_ratings.py`: extract responses from jsonl file retrieved from OPENAI server
    - `eval_ratings.py`: get the average ratings for each test configurations (e.g., sht, raptor, bm25, etc.)


# Eexperiment Results

- `database/`: The constructed SHTs
- `indexes/`: the indexes of the nodes w.r.t. the queries
- `contexts/`: the contexts of the queries
- `queries/`: the queries together with its gold answers and promp templates
- `llm_judge/qasper`: the ratings

    `llm_judge/qasper-databricks` is the ratings created by using the old propmt (databrick as the example)

- `tmp/comphrdoc`: show that c-strongly-templatized documents are common
- `answers/`: the answers of different dataset and different test configurations. The statistics of the answers are 
    - `answers/civic-answer-num.json`
    - `answers/civic2-answer-recall-precision-f1.json`
    - `answers/contract-answer-num-llm-gold.json` (this is when the gold answer is the llm answer given the full document)
    - `answers/contract-answer-num.json`
    - `answers/qasper-answer-f1-gold.json` (this when the context for QA is the gold context)
    - `answers/qasper-answer-f1.json`
    - `answers/qasper-answer-llm-rating-databricks.json` (this is when the judging prompt is the old databrick prompt)
    - `answers/qasper-answer-llm-rating.json`

- `answer_contest_stats/contract`: the jaccard similarity of the retrieved context compared with the gold context

- `wrong_answers/`: the wrong answers