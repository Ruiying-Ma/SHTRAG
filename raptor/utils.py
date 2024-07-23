import logging
import re
from typing import Dict, List, Set

import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens. Sentences in the chunks can all be directly matched in the intput text.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
    
    Returns:
        List[str]: A list of text chunks.
    """

    def split_text_into_sentences(delimiters, text):
        '''
        Split the text into sentences using multiple delimiters. Each sentence can be found in the original text.
        '''
        regex_pattern = f"([{re.escape(''.join(delimiters))}])"
        split_parts = re.split(regex_pattern, text)
        sentences = [split_parts[i] + split_parts[i + 1] for i in range(0, len(split_parts) - 1, 2) if len(split_parts[i].strip()) > 0]
        if len(split_parts) % 2 != 0 and len(split_parts[-1].strip()) > 0:
            sentences.append(split_parts[-1])
        return sentences

    sentence_delimiters = [".", "!", "?", "\n"]
    sentences = split_text_into_sentences(sentence_delimiters, text)
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue

        if current_length + token_count <= max_tokens:
            current_chunk.append(sentence)
            current_length += token_count
        else: 
            # current_length + token_count > max_tokens
            if current_chunk:
                chunks.append("".join(current_chunk))
            if token_count <= max_tokens:
                current_chunk = [sentence]
                current_length = token_count
            else:
                # token_count > max_tokens
                chunks.append(sentence)
                current_chunk = []
                current_length = 0

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)
