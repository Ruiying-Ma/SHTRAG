import re
from typing import Dict, List
import string

def max_min(list: List[int], thresh: int):
    '''
    Return the maximal item in list that is smaller than thresh. If list is empty, return 0

    Args:

        - list (List[int])

        - thresh (int)

    Return:

        - the maximal item in list that is smaller than thresh
    '''
    if (not isinstance(list, List)) or (not all([isinstance(l, int) for l in list])):
        raise ValueError("list must be a list of integers")
    if not isinstance(thresh, int):
        raise ValueError("thresh must be an integer")
    if thresh < 1:
        raise ValueError("thresh must be >= 1")
    if any([item < 1 for item in list]):
        raise ValueError("list must be a list of integers that are all >= 1")

    min_list = [item for item in list if item < thresh]
    if len(min_list) == 0:
        return 0

    return max(min_list)

def clean_space(text: str) -> str:
    return " ".join(re.sub(r'\s', " ", text).split())

def split_text_into_sentences(delimiters, text) -> List:
    '''
    Split the text into sentences using multiple delimiters. Each sentence can be found in the original text.

    Args:

        - delimiters

        - text

    Return:

        - A list of sentences
    '''
    regex_pattern = f"([{re.escape(''.join(delimiters))}])"
    split_parts = re.split(regex_pattern, text)
    sentences = [split_parts[i] + split_parts[i + 1] for i in range(0, len(split_parts) - 1, 2) if len(split_parts[i].strip()) > 0]
    if len(split_parts) % 2 != 0 and len(split_parts[-1].strip()) > 0:
        sentences.append(split_parts[-1])
    return sentences

def split_text_into_chunks(chunk_size, text, tokenizer) -> List:
    '''
    Split the text into sentences, then concat sentences into chunks. The concatenator is "".

    Args:

        - chunk_size: counted by token num

        - text

        - tokenizer

    Return:

        - A list of chunks
    '''
    sentence_delimiters = [".", "!", "?", "\n"]
    sentences = split_text_into_sentences(sentence_delimiters, text)
    n_tokens = [len(tokenizer.encode(sentence)) for sentence in sentences]

    chunks = [] # list of chunks
    current_chunk = [] # list of sentences
    current_length = 0 # current chunk size

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue

        if current_length + token_count <= chunk_size:
            current_chunk.append(sentence)
            current_length += token_count
        else: 
            # current_length + token_count > chunk_size
            if len(current_chunk) > 0:
                chunks.append("".join(current_chunk))
            if token_count <= chunk_size:
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

def get_nondummy_ancestors(nodes: List[Dict], node_id: int) -> List[int]:
    '''
    Get the list of nondummy ancestors' ids of the given nondummy node. Return the ids in increasing order. Note that -1 is not contained.
    '''

    if not isinstance(node_id, int) or node_id < 0 or node_id >= len(nodes):
        raise ValueError("node doesn't exist")
    
    node = nodes[node_id]

    if node["is_dummy"]:
        raise ValueError("node is dummy")

    if "nondummy_parent" not in node:
        raise ValueError("Must execute self._add_nondummy_parent() in advance")

    cur_parent_id = node["nondummy_parent"]
    ancestors = []
    while cur_parent_id != -1:
        ancestors.append(cur_parent_id)
        cur_node = nodes[cur_parent_id]
        assert not cur_node["is_dummy"]
        cur_parent_id = cur_node["nondummy_parent"]

    assert ancestors[::-1] == sorted(ancestors)
    if len(ancestors) > 0:
        assert ancestors[0] == node["nondummy_parent"]
    return ancestors[::-1]