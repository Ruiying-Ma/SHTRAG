# Sound SHT Extraction

## Step 1: Heading Identification
Use VGT (first start VGT in docker):
```
> curl -X POST -F 'file=@{path_to_pdf}' localhost:5060
```

## Step 2: Node Clustering
### FeatureExtractor
file: [FeatureExtractor.py](./FeatureExtractor.py)

classes:
- `FeatureExtractor`: Extract the visual patterns of the headings and the list items extracted by VGT.
    - For the headings, the extracted visual patterns are: `font_size` (rounded to 2), `font_name`, `font_color`, `is_all_cap` (alphabetic characters only), `is_centered` ($|mid_{bbox} - mid_{page}| \leq 2$), `list_type`, `is_underlined`.
    - For the list items, the extracted visual patterns are: the corresponding heading, `list_type`.

### ClusteringOracle
file: [ClusteringOracle.py](./ClusteringOracle.py)

classes:
- `ClusteringOracleConfig`: Set the configurations, including the constant parameters for `FeatureExtractor`.
- `ClusteringOracle`: Cluster the nodes (corresponding to the headings and the list items) according to their visual patterns.

## Step 3: Tree Construction
file: [SHTBuilder.py](./SHTBuilder.py)

classes: 
- `SHTBuilderConfig`: Set the SHT configurations, including the chunk size, the summary length, the embedding model, and the summarization model.
- `SHTBuilder`: 
    - `build()`: Build the SHT.


# StructuredRAG
## Step 4: Integration of SHT into Structured-RAG
file: [SHTBuilder.py](./SHTBuilder.py)

- `SHTBuilder`: 
    - `build(): `Add new leaf nodes, storing the chunks. Populate the heading attributes of the nodes.
    - `add_summaries()`: Populate the context attributes of the nodes. 
    - `add_embeddings()`: Embed the nodes into high-dimensional space.

## Step 5: Nodes Retrieval
file: [SHTIndexer.py](./SHTIndexer.py)

- `SHTIndexerConfig`: Set the retriever's configurations, including whether to use the embedding that contains the hierarchical information, the embedding model for the query, and the distance metric in the embedding space.
- `SHTIndexer`: Sort the SHT nodes in ascending order of their distances to the query.

## Step 6: Context Generation
file: [SHTGenerator.py](./SHTGenerator.py)

- `SHTGeneratorConfig`: Set the generator's configurations, including whether to recover the hierarchical structures in the final context, whether to retrieve the contexts of the newly added leaves in Step 4, and the length of the final context.
- `SHTGenerator`: Generate the final context using the retrieved nodes.

