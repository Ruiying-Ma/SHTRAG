import json
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
from sht import split_text_into_chunks
import tiktoken

from sht import SHTBuilder, SHTBuilderConfig
import os

def add_node(type, parent, text, nodes, m_height_to_ids_list, m_id_to_height, full_texts_list):
    assert text != ""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    if type == "head":
        heading = text
        texts = [""]
        embeddings = {
            "heading": None,
            "texts": None,
            "hybrid": None,
        }
    else:
        assert type == "text"
        heading = ""
        texts = split_text_into_chunks(100, text, tokenizer)
        embeddings = {
            "texts": None,
            "hybrid": None,
        }

    new_node = {
        "is_dummy": False,
        "id": len(nodes),
        "type": type,
        "parent": parent,
        "children": [],
        "nondummy_parent": parent,
        "nondummy_children": [],
        "heading": heading,
        "texts": texts,
        "embeddings": embeddings,
        "info": {
            "features": None,
            "cluster_id": parent + 1,
        }
    }

    nodes.append(new_node)
    assert parent in m_id_to_height
    height = m_id_to_height[parent] + 1
    if height not in m_height_to_ids_list:
        m_height_to_ids_list[height] = []
    m_height_to_ids_list[height].append(new_node["id"])
    m_id_to_height[new_node["id"]] = height
    full_texts_list.append(text)


def build_sht():
    '''
    cluster_id = parent + 1
    '''
    with open("/mnt/f/Research-2024-spring/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json", 'r') as file:
        data = json.load(file)

    for file_name in list(data.keys()):
        print(file_name)
        doc_info = data[file_name]
        nodes = []
        m_height_to_ids_list = {-1: [-1]}
        m_id_to_height = {-1: -1}
        estimated_cost = {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        full_texts_list = []

        # title
        assert doc_info["title"] != ""
        add_node(type="head", parent=-1, text=doc_info["title"], nodes=nodes, m_height_to_ids_list=m_height_to_ids_list, m_id_to_height=m_id_to_height, full_texts_list=full_texts_list)
        # abstract
        if doc_info["abstract"] != "":
            add_node(type="head", parent=0, text="Abstract", nodes=nodes, m_height_to_ids_list=m_height_to_ids_list, m_id_to_height=m_id_to_height, full_texts_list=full_texts_list)
        # body
        m_sect_name_node_id = dict()
        for section in doc_info["full_text"]:
            # heading
            assert len(set(m_sect_name_node_id.values())) == len(m_sect_name_node_id)
            if section["section_name"] is None or section["section_name"] == "":
                raw_names = [""]
            else:
                raw_names = section["section_name"].split(":::")
            assert len(raw_names) > 0
            raw_sect_names = [n.strip() for n in raw_names]
            sect_names = []
            for s in raw_sect_names:
                if s == "":
                    sect_names.append(" ")
                else:
                    sect_names.append(s)
            parent = 0
            section_id = None
            first_new_sect_name_index = 0
            for sid, s in enumerate(sect_names):
                s_key = " ::: ".join(sect_names[:(sid + 1)])
                if s_key in m_sect_name_node_id:
                    parent = m_sect_name_node_id[s_key]
                    continue
                else:
                    first_new_sect_name_index = sid
                    break
            
            assert first_new_sect_name_index < len(sect_names)
            for i in range(first_new_sect_name_index, len(sect_names)):
                key = " ::: ".join(sect_names[:(i + 1)])
                assert key not in m_sect_name_node_id
                section_id = len(nodes)
                add_node(type="head", parent=parent, text=sect_names[i], nodes=nodes, m_height_to_ids_list=m_height_to_ids_list, m_id_to_height=m_id_to_height, full_texts_list=full_texts_list)
                parent = section_id
                assert i + 1 == m_id_to_height[section_id]
                m_sect_name_node_id[key] = section_id
            # paragraphs
            text = "\n\n".join([p for p in section["paragraphs"]])
            if text == "":
                continue
            assert text != ""
            assert section_id != None
            parent = section_id
            add_node(type="text", parent=parent, text=text, nodes=nodes, m_height_to_ids_list=m_height_to_ids_list, m_id_to_height=m_id_to_height, full_texts_list=full_texts_list)
        
        # set children for each node
        for node in nodes:
            if node["parent"] != -1:
                parent = nodes[node["parent"]]
                parent["children"].append(node["id"])

        # set nondummy_children for each node
        for node in nodes:
            node["nondummy_children"] = node["children"]

        # estimate cost
        tokenizer = tiktoken.get_encoding("cl100k_base")
        for node in nodes:
            if node["type"] == "text":
                tokens = len(tokenizer.encode("\n\n".join(node["texts"]) + "\n\n"))
            else:
                tokens = len(tokenizer.encode(node["heading"] + "\n\n")) + 100
            
            if node["parent"] != -1:
                estimated_cost["input_tokens"] += tokens
            if len(node["children"]) > 0:
                estimated_cost["output_tokens"] += tokens

        tree = {
            "nodes": nodes,
            "m_height_to_ids_list": m_height_to_ids_list,
            "m_id_to_height": m_id_to_height,
            "full_text": "\n\n".join(full_texts_list),
            "estimated_cost": estimated_cost
        }
        with open(f"/mnt/f/Research-2024-spring/SHTRAG/database/qasper/sht/{file_name}.json", 'w') as file:
            json.dump(tree, file, indent=4)


def test_sht_builder_with_load():
    in_json_dir = "/mnt/f/Research-2024-spring/SHTRAG/database/qasper/sht"
    out_dir = "/mnt/f/Research-2024-spring/SHTRAG/database/qasper/sht"
    files = os.listdir(in_json_dir)
    
    for json_file in files:
        if not json_file.endswith(".json"):
            continue

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        
        load_json = os.path.join("/mnt/f/Research-2024-spring/SHTRAG/database/qasper/sht", json_name)
        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=load_json, chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(None)
        # sht_builder.store2json()
        sht_builder.visualize()

def test_qasper_sht_builder():
    build_sht()
    test_sht_builder_with_load()





if __name__ == "__main__":
    test_qasper_sht_builder()


