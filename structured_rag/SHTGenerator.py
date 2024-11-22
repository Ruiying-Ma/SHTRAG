import logging
from typing import Dict, List
import tiktoken

from .utils import get_nondummy_ancestors



class SHTGeneratorConfig:
    def __init__(self, use_hierarchy: bool, use_raw_chunks: bool, context_len: int):
        '''
        Args:
            - use_hierarchy (bool): whether add hierarchy to final context
            - context_len (int): the maximal number of tokens of the generated context
        '''
        if not isinstance(use_hierarchy, bool):
            raise ValueError("use_hierarchy must be a boolean value")

        if not isinstance(use_raw_chunks, bool):
            raise ValueError("use_raw_chunks must be a boolean value")

        if not isinstance(context_len, int) or context_len < 0:
            raise ValueError("context_len must be a nonnegative integer")
        
        self.use_hierarhy = use_hierarchy
        self.use_raw_chunks = use_raw_chunks
        self.context_len = context_len


            
class SHTGenerator:
    def __init__(self, config: SHTGeneratorConfig):
        if not isinstance(config, SHTGeneratorConfig):
            raise ValueError("config must be an instance of SHTGeneratorConfig")

        self.use_hierarchy = config.use_hierarhy
        self.use_raw_chunks = config.use_raw_chunks
        self.context_len = config.context_len
        self.tokenizer = tiktoken.get_encoding("cl100k_base")


    def generate(self, candid_indexes: List[Dict], nodes: List[Dict]) -> str:
        '''
        Return the generated context
        '''
        log_info = "structured context" if self.use_hierarchy else "raw context"
        log_info += " with raw chunks" if self.use_raw_chunks else " without raw chunks"
        logging.info(f"Generating {log_info}...")

        assert all([not nodes[i["node_id"]]["is_dummy"] for i in candid_indexes])

        if self.use_raw_chunks:
            indexes = candid_indexes
        else:
            indexes = [i for i in candid_indexes if nodes[i["node_id"]]["type"] != "text"]

        if not self.use_hierarchy:
            token_count = 0
            context = ""
            for index in indexes:
                node_id = index["node_id"]
                chunk_id = index["chunk_id"]

                node = nodes[node_id]
                assert not node["is_dummy"]
                if not self.use_raw_chunks:
                    assert node["type"] != "text"
                heading_string = node["heading"]
                if heading_string != "":
                    heading_string += "\n\n"
                assert chunk_id >= 0 and chunk_id < len(node["texts"])
                text_string = node["texts"][chunk_id]
                if text_string != "":
                    text_string += "\n\n"

                text = heading_string + text_string
                text_len = len(self.tokenizer.encode(text))
                if text_len + token_count <= self.context_len:
                    context += text
                    token_count += text_len
                else:
                    break
            
            assert len(self.tokenizer.encode(context)) <= self.context_len
            assert len(self.tokenizer.encode(context)) == token_count
            return context

        # use hierarchy
        assert self.use_hierarchy
        candid_nodes_for_structure = [] # nodes whose headings will be added; must be head-/list-type nodes. 
        candid_nodes_for_text = dict() # nodes whose texts will be added. A map of node_id to a set of chunks
        token_count = 0
        for index in indexes:
            node_id = index["node_id"]
            chunk_id = index["chunk_id"]

            node = nodes[node_id]

            if not self.use_raw_chunks:
                assert node['type'] != "text"

            ancestor_ids = get_nondummy_ancestors(nodes, node_id)
            if node["type"] != "text":
                ancestor_ids.append(node_id)

            new_candid_nodes_for_structure = []
            new_token_count = 0

            assert chunk_id >= 0 and chunk_id < len(node["texts"])
            new_text = node["texts"][chunk_id]
            if new_text != "":
                new_text += "\n\n"
            new_token_count += len(self.tokenizer.encode(new_text))

            if len(ancestor_ids) > 0:
                assert all([not nodes[aid]["is_dummy"] for aid in ancestor_ids])
                assert all([nodes[aid]["type"] in ["head", "list"] for aid in ancestor_ids])
                assert sorted(ancestor_ids) == ancestor_ids
                i = 0
                while i < len(ancestor_ids):
                    if ancestor_ids[i] in candid_nodes_for_structure:
                        i += 1
                    else:
                        break
                if i < len(ancestor_ids):
                    assert [aid not in candid_nodes_for_structure for aid in ancestor_ids[i:]]
                    for aid in ancestor_ids[i:]:
                        new_candid_nodes_for_structure.append(aid)
                        new_heading = nodes[aid]["heading"]
                        assert new_heading != ""
                        new_heading += "\n\n"
                        new_token_count += len(self.tokenizer.encode(new_heading))
            
            if token_count + new_token_count <= self.context_len:
                candid_nodes_for_structure += new_candid_nodes_for_structure
                token_count += new_token_count
                if node_id not in candid_nodes_for_text:
                    candid_nodes_for_text[node_id] = set()
                else:
                    assert chunk_id not in candid_nodes_for_text[node_id]
                candid_nodes_for_text[node_id].add(chunk_id)
            else:
                break

        assert token_count <= self.context_len
        candid_nodes = sorted(list(set(candid_nodes_for_structure).union(set(candid_nodes_for_text.keys()))))

        context = ""
        for nid in candid_nodes:
            candid_node = nodes[nid]
            assert not candid_node["is_dummy"]
            heading_string = candid_node["heading"]
            if heading_string != "":
                heading_string += "\n\n"

            text_string = ""
            if nid in candid_nodes_for_text:
                chunks = sorted(list(candid_nodes_for_text[nid]))
                for cid in chunks:
                    assert cid >= 0 and cid < len(candid_node["texts"])
                    chunk_text = candid_node["texts"][cid]
                    if chunk_text != "":
                        chunk_text += "\n\n"
                    text_string += chunk_text

            context += heading_string + text_string

        assert len(self.tokenizer.encode(context)) == token_count
        assert len(self.tokenizer.encode(context)) <= self.context_len
        return context


        

                