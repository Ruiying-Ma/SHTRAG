# SHTRAG pipeline
## Components
`feature_extractor.py`: extract visual patterns
- `FeatureExtractor`

`clustering_oracle.py`: cluster headings according to visual patterns extracted by FeatureExtractor
- `ClusteringOracleConfig`
- `ClutseringOracle`

`sht_builder.py`: build the sht - summarization and embedding
- `SHTBuilderConfig`
- `SHTBuilder`


`sht_indexer.py`: index the nodes and sort them by priority
- `SHTIndexerConfig`
- `SHTIndexer`

`sht_generator.py`: generate the context given the retrieved nodes
- `SHTGeneratorConfig`
- `SHTGenerator` 

`SummarizationModels.py`: create summaries

`EmbeddingModels.py`: create embeddings

## `FeatureExtractor` (`feature_extractor.py`)
Process the results extracted by `pdf-document-layout-analysis` -- extract features of each title/section header/list item. The extracted features are appended (rewritten) to the original json file returned by `pdf-document-layout-analysis`. 

Features:
- For "Title" and "Section header": 
    - `font_size` (rounded to 2)
    - `font_name`
    - `font_color`
    - `is_all_cap`

        Only alphabetic characters
    - `is_centered`

        Whether the mid of the bbox $\approx$ mid of the page with error = 2
    - `list_type`

        TEMPLATE: 
        - `(a)`
        - `a.`
        - `a)`
        - `§a`
        - `ARTICLE `, `ARTICLE a`
        - `article `, `article a`
        - `SECTION `, `SECTION a`
        - `section `, `section a`
        - `ITEM `, `ITEM a`
        - `item `, `item a`

        Types for `a`: 
        - numeric
        - roman_upper
        - romman_lower
        - alpha_upper
        - alpha_lower
        - numeric_multilevel
        - bullet
    - `is_underlined`

        whether the begin and end of a bbox has an aligned line
- For "List item":
    - predecessor "Title"/"Section header"'s id
    - `list_type`

### Initialize a FeatureExtractor
```python
feature_extractor = FeatureExtractor(
    thresh_is_centered=config.thresh_is_centered, # the threshold of judging whether a bbox is centered, default as 2.0
    round_font_size=config.round_font_size, # the number of digits rounded for font size, default as 2
    thresh_is_underlined=config.thresh_is_underlined, # the threshhold of judging whether a bbox is underlined, default as 6.0
    thresh_is_line=config.thresh_is_line, # the threshold of deciding whether a fitz.Rect can be recognized as a line, default as 2.0
    thresh_rect=config.thresh_rect, # added to the border of the bbox returned by pdf-dpcument-layout-analysis, default as 0.0
)
```

### Use a FeatureExtractor
If you want to extract features of a bbox `b`:
1. Load the feature extractor with the target pdf:
    ```python
    feature_extractor.load_pdf_doc(pdf_path) # pdf_path: absolute path to the target pdf file
    ```
2. Laod the page where `b` is in:
    ```python
    feature_extractor.load_page_and_underlines(page_number) # page_number: 1-based integer
    ```
3. Load the bbox `b`:
    ```python
    feature_extractor.load_rect(
        left=b["left"], # distance between: page's left margin & bbox's left margin
        top=b["top"], # distance between: page's top margin & bbox's top margin
        width=b["width"], # width of the bbox
        height=b["height"], # height of the bbox
        page_number=page_number, # 1-based integer, where b is in
    )
    ```
4. Extract features except the list type:
    1. font information (font size, name, color)
        ```python
        feature_info = feature_extractor.extract_font_info()
        ```
    2. is_centered
        ```python
        is_centered = feature_extractor.is_centered()
        ```
    3. is_underlined
        ```python
        is_underlined = feature_extractor.is_unnderlined()
        ```
    4. is_all_cap
        ```python
        is_all_cap = feature_exrtactor.is_all_cap(text) # text: the text contained in the bbox
        ```
5. After all bboxes finish step 4, their list types are extracted
    ```python
    list_types_list = feauter_extractor.extract_list_type(texts_list) # texts_list: the list of texts of the list of the bboxes. THe order follows the reading order.
    ```

### Example usage
See ClusteringOracle.cluster() in `clustering_oracle.py`

## `ClusteringOracle` (`clustering_oracle.py`)
Use a FeatureExtractore to cluster the objects returned by `pdf-document-layout-analysis`

### Initialize a ClusteringOracle
1. Set the config
    ```python
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=None, # the absolute path to a json file to store the result; if set None, then don't use store2json() method,
    )
    ```
2. Create an instance
    ```python
    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    ```

### Use a ClusteringOracle
1. Cluster the objects returned by `pdf-document-layout-analysis` using FeatureExtractor
    ```python
    new_object_dict_lists = clustering_oracle.cluster(pdf_path, object_dicts_list) # pdf_path: the absolute path to the target pdf file; object_dicts_list: the result of pdf-document-layout-analysis
    ```

### Example usage
See `test_clustering_oracle()` in `SHTRAG/test/test.py`. If you want to store the new_object_dict_lists in json form, set `store_json` as an absolute path to the destination json file.

## `SHTBuilder` (`sht_builder.py`)
Build the SHT based on the result returned by the clutsering_oracle (new_object_list_dicts).
### Initialize an SHTBuilder
1. Set the config
    ```python
    sht_builder_config = SHTBuilderConfig(
        store_json=os.path.join(out_dir, json_name), # store_json: must be an absolute path to a json file (even if you don't want to store the result);
        load_json=None, # if not None, load_json is the absolute path to the json file that stores an SHT; otherwise, None
        chunk_size=100, # the token limit of the raw test chunks, default to be 100
        summary_len=100, # the token limit of the summaries of heading nodes, default to be 100
        embedding_model_name=embedding_mode_name, # the name of the embedding model, can be chosen from `dpr`, `sbert`, and `te3small` (test-embedding-3-small)
        openai_key_path=path, # path to the file that stores openai_key
    )
    ```
2. Create an instance
    ```python
    sht_builder = SHTBuilder(sht_builder_config)
    ```
### Use an SHTBuilder
1. Build SHT: create the tree shape; summaries are embeddings are not added
    If `sht_builder.load_json` is None, build from scratch
    ```python
    sht_builder.build(new_object_dicts_list) # new_object_dicts_list is the result returned by ClutseringOracle
    ```
    If `sht_builder.load_json` is not None, the SHT is directly loaded from the given file.
    ```python
    sht_builder.build(None)
    ```
2. Add summaries to heading nodes
    ```python
    stats = sht_builder.add_summaries()
    ```
    stats is the time and token counts of adding summaries
3. Add embeddings to all tree nodes
    ```python
    stats = sht_builder.add_embeddings(node_ids_list) # node_ids_list: the ids of the nodes to add embeddings
    ```
    stats is the time of adding embeddings
    - hybrid: embedding time with hierarchical information
    - text: embedding time without hierarchical information
    - heading: embedding time of the headings
4. Store SHT in json format
    ```python
    sht_builder.store2json()
    ```
    The destination path is `sht_builder.store_json`
5. Visualize SHT and store it
    ```python
    sht_builder.visualize()
    ```
    The destination path is `sht_builder.store_json.replace(".json", ".txt")`

### Example usage

See `test_sht_builder()` in `SHTRAG/test/test.py` (build SHT from scratch) as an example for step 1

See `test_sht_builder_with_load()` in `SHTRAG/test/test.py` (load already-existed SHT) as an example for step 1

See `sht_build_add_summary_with_load()` in `SHTRAG/test/test.py` as an example for step 1-2

See `sht_builder_add_embeddings()` in `SHTRAG/test/test.py`as an example for step 1-3. Note that before executing `add_emebddings()`, you should make sure that the summaries have already been created.

Note that there may be errors in the example code (due to code updates). The goal is to show the pipeline. 

## `SHTIndexer` (`sht_indexer.py`)
Given a fully-constucted (build, add summaries and embeddings) SHT, index its nodes and return the node indexes in order of priority.

### Intialize an SHTIndexer
1. Set the configs
    ```python
    indexer_config = SHTIndexerConfig(use_hierarchy=use_hierarchy, # whether hierarchical information should be included in the embedding. If so, True and use hybrid_emebdding; If not, False and use texts_embedding
    distance_metric="cosine", # distance metric
    query_embedding_model_name=embedding_model), # embedding model of the query. For SBERT and text-embedding-3-small, the emebedding models for the nodes and the query are the same; for DPR, the nodes are embedded with a context encoder, while the query is embedded with a query encoder
    ```

2. Create an instance
    ```python
    indexer = SHTIndexer(indexer_config)
    ```

### Use an SHTIndexer
1. Index the SHT nodes
    ```python
    indexes = indexer.index(query, nodes) # query is a string; nodes denote the SHT (dictionaries)
    ```

### Example usage
See `index_for_one_dataset_sht()` in `SHTRAG/test/test_indexer.py`

Note that BM25 for SHTRAG is a different. It doesn't have embedding process in SHTBuilder. It only indexes nodes and generates context. See `bm25_indexing_for_sht_and_shtr()` in `SHTRAG/test/test_bm25.py`.

## `SHTGenerator` (`test_generator.py`)
Given the indexes of the SHT nodes sorted in priority order, return the context within token limit

### Initialize an SHTGenerator
1. Set the configs
    ```python
    generator_config = SHTGeneratorConfig(use_hierarchy=generator_use_hierarchy, # whether the retrieved nodes are organized using the SHT structure. If not, they are directly connected in priority order (not reading order)
    use_raw_chunks=generator_use_raw_chunks, # whether the raw text chunks in the SHT should be included in the final context
    context_len=context_len) # the token limit of the final context
    ```
2. Creat an instance
    ```python
    generator = SHTGenerator(generator_config)
    ```
### Use an SHTGenerator
1. Create the context
    ```python
    context = generator.generate(indexes, nodes) # indexes: the list of node ids sorted in priority order (i.e., the result returned by SHTIndexer); nodes: the SHT
    ```

### Example usage
See `generate_for_one_dataset_sht()` in `SHTRAG/test/sht_builder.py`

