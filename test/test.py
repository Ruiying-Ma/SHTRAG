import json
import os
import sys
sys.path.append("/mnt/f/Research-2024-spring/SHTRAG")
import numpy as np

def test_split_text():
    import tiktoken
    from raptor import split_text
    text = "Hello world! How are you today  ? This is an example.\nNew line here: airport at 10:30 is so good. New line here: test. 1.2.3.4.5.6,7,8,9,10,11,12"
    chunks = split_text(text=text, max_tokens=6, tokenizer=tiktoken.get_encoding("cl100k_base"))
    print(chunks)

def test_summarization_model():
    from sht import GPT4oMiniSummarizationModel

    text = '''I. The parties desire to have discussions of or relating to the Subject Matter for the purposes of evaluating a possible
business relationship between them (“Purpose”). The parties may extend the Subject Matter or add additional
parties by executing one or more addenda to this Agreement.
II. Such discussions may involve disclosure by one party to the other party of confidential, proprietary or trade secret
information of its own or its licensors (“Confidential Information” as defined below), during the Period for Exchange
of Information.
III. Both parties recognize the value of the Confidential Information and that it is in their mutual best interests to
maintain the confidential, proprietary and secret nature of the Confidential Information.
THEREFORE, in consideration of the Subject Matter, and the mutual promises herein, the parties agree as follows:
1. CONFIDENTIAL INFORMATION. The term “Confidential Information” as used herein means all nonpublic
information relating to the Subject Matter that is disclosed by either party, its Affiliates (as defined below), or their
agents (where applicable, collectively referred to as the “Disclosing Party”), directly or indirectly, in writing, orally
or by inspection of premises or tangible objects to the other party (the “Recipient”) that is: (i) marked
confidential or proprietary, or (ii) given the nature of the information or the circumstances surrounding its disclosure,
reasonably should be deemed confidential. Confidential Information includes, but is not limited to documents,
drawings, models, apparatus, sketches, designs, schedules, product plans, marketing plans, technical procedures,
manufacturing processes, software, prototypes, samples, methodologies, formulations, trade secrets, patent
applications, know-how, experimental results, specifications and other business information.
2. PERIOD OF CONFIDENTIALITY AND NON-USE. The Recipient will use Confidential Information only in
connection with the Purpose as set forth in this Agreement. Recipient shall use the same degree of care to avoid
disclosure or use of the Confidential Information as it uses for its own confidential, proprietary and trade secret
information, but in no case use less than a reasonable degree of care. Recipient agrees to limit disclosure of
Confidential Information to employees and employees of Affiliates having a specific need to know such Confidential
Information for the Purpose and in the case of Affiliates only to the extent that such Affiliate is under obligation to
hold such information in confidence and is made aware of these terms and conditions. Recipient will not disclose or
permit access to Confidential Information to contract workers, consultants or contractors of Recipient or its Affiliates
unless authorized by Disclosing Party in writing and on condition that such persons are bound by obligations of
confidentiality inuring to the benefit of Disclosing Party and its Affiliates at least as restrictive as these terms and
conditions. Recipient shall not without Disclosing Party’s prior written consent reverse engineer, disassemble or
decompile any prototypes, software or other objects which embody the Disclosing Party’s Confidential Information
to obtain access to Disclosing Party’s trade secrets and to the extent such consent is granted Recipient shall
receive and hold such Confidential Information subject to the terms of this Agreement.
'''

    summary_len = 100

    summarizer = GPT4oMiniSummarizationModel(openai_key_path="/home/ruiying/Documents/Codebase/config/openai/config_openai.txt")

    summary_response = summarizer.summarize(text, summary_len)
    
    print(summary_response)

def test_clustering_oracle():
    from sht import ClusteringOracle, ClusteringOracleConfig
    pdf_dir = "/mnt/f/Research-2024-spring/financebench/pdfs"
    in_dir = "/mnt/f/Research-2024-spring/RetrieveSys/data/adobe_extract/finance"
    # out_dir = "/mnt/f/Research-2024-spring/SHTRAG/database/civic/sht"
    files = os.listdir(in_dir)
    count = 0
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=None,
    )
    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    for json_file in files:
        if not json_file.endswith(".json"):
            continue
        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        pdf_name = f"{file_name}.pdf"
        
        with open(os.path.join(in_dir, json_name), 'r') as file:
            object_dicts_list = json.load(file)
        clustering_oracle.cluster(pdf_path=os.path.join(pdf_dir, pdf_name), object_dicts_list=object_dicts_list)
        count += 1

    print(count)

def test_clustering_oracle_performance_with_comphrdoc():
    from sht import SHTBuilder, SHTBuilderConfig, ClusteringOracle, ClusteringOracleConfig
    pdf_dir = "/mnt/f/Research-2024-spring/SHTgen/experiments/comphrdoc/test_eval_pdf"
    gold_path = "/mnt/f/Research-2024-spring/SHTgen/experiments/results/comphrdoc/gold_comphrdoc_sht.json"
    out_dir = "/mnt/f/Research-2024-spring/SHTRAG/tmp/comphrdoc"
    with open(gold_path, 'r') as file:
        data = json.load(file)
    m_file_to_stats = dict()
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=None,
    )
    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    for json_file in list(data.keys()):
        if not json_file.endswith(".json"):
            continue

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        pdf_name = f"{file_name}.pdf"
        
        gold_object_dicts_list = data[json_file]

        object_dicts_list = []

        for gold_object_dict in gold_object_dicts_list:
            if gold_object_dict["relation"] == "connect":
                continue
            object_dicts_list.append({
                "left": float(gold_object_dict["box"][0]),
                "top": float(gold_object_dict["box"][1]),
                "width": float(gold_object_dict["box"][2] - gold_object_dict["box"][0]),
                "height": float(gold_object_dict["box"][3] - gold_object_dict["box"][1]),
                "page_number": gold_object_dict["page"]+1,
                "page_width": 0,
                "page_height": 0,
                "text": gold_object_dict["text"],
                "type": "Title"
            })

        new_object_dicts_list = clustering_oracle.cluster(pdf_path=os.path.join(pdf_dir, pdf_name), object_dicts_list=object_dicts_list)
        
        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=None, chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(new_object_dicts_list)
        sht_builder.store2json()
        sht_builder.visualize()
        # stats
        nondummy_nodes = [n for n in sht_builder.tree["nodes"] if not n["is_dummy"]]
        assert len(nondummy_nodes) == len(object_dicts_list)
        gold_nodes = [o for o in gold_object_dicts_list if o["relation"] != "connect"]
        assert len(nondummy_nodes) == len(gold_nodes)
        m_id_to_gold = {
            nondummy_nodes[i]["id"]: gold_nodes[i]["id"]
            for i in range(len(nondummy_nodes))
        }
        num_correct_ancestor_nodes = 0
        num_perfect_ancestor_nodes = 0
        for i in range(len(nondummy_nodes)):
            node = nondummy_nodes[i]
            gold_node = gold_nodes[i]
            ancestors = sht_builder._get_nondummy_ancestors(node["id"])
            gold_ancestors = gold_node["ancestors"][1:]
            assert gold_ancestors == sorted(gold_ancestors)
            assert all([a in set(m_id_to_gold.keys()) for a in ancestors])
            assert all([a in set(m_id_to_gold.values()) for a in gold_ancestors])

            m_ancestors = [m_id_to_gold[a] for a in ancestors]
            assert m_ancestors == sorted(m_ancestors)

            print(gold_node["text"], ancestors, gold_ancestors, m_ancestors)########################
            if set(gold_ancestors).issubset(set(m_ancestors)):
                num_correct_ancestor_nodes += 1
            if set(gold_ancestors) == set(m_ancestors):
                assert gold_ancestors == m_ancestors
                num_perfect_ancestor_nodes += 1

        is_perfect_tree = (num_perfect_ancestor_nodes == len(nondummy_nodes))
        is_correct_tree = (num_correct_ancestor_nodes == len(nondummy_nodes))
        

        num_correct_ts_nodes = 0
        num_perfect_ts_nodes = 0
        for i in range(len(nondummy_nodes)):
            node = nondummy_nodes[i]
            gold_node = gold_nodes[i]
            ts = sht_builder._get_nondummy_successor(node["id"])
            gold_ts = gold_node["textspan"]
            if ts != sys.maxsize:
                assert ts in set(m_id_to_gold.keys())
            if gold_ts != sys.maxsize:
                assert gold_ts in set(m_id_to_gold.values())
            if ts == sys.maxsize:
                m_ts = ts
            else:
                m_ts = m_id_to_gold[ts]

            if m_ts >= gold_ts:
                num_correct_ts_nodes += 1
            if m_ts == gold_ts:
                num_perfect_ts_nodes += 1

        assert (num_correct_ts_nodes == len(nondummy_nodes)) == is_correct_tree
        # strong-templatization
        m_gold_to_id = {
            gold_nodes[i]["id"]: nondummy_nodes[i]["id"]
            for i in range(len(nondummy_nodes))
        }
        strongly_templatized = True
        m_cluster_id_to_node_ids = dict()
        for nid in list(m_gold_to_id.values()):
            cluster_id = sht_builder.tree["nodes"][nid]["info"]["cluster_id"]
            if cluster_id not in m_cluster_id_to_node_ids:
                m_cluster_id_to_node_ids[cluster_id] = []
            m_cluster_id_to_node_ids[cluster_id].append(nid)
        for cluster_id in m_cluster_id_to_node_ids:
            assert m_cluster_id_to_node_ids[cluster_id] == sorted(m_cluster_id_to_node_ids[cluster_id])
            ancestor_lists = []
            for nid in m_cluster_id_to_node_ids[cluster_id]:
                gold_nid = m_id_to_gold[nid]
                ancestors = gold_object_dicts_list[gold_nid]["ancestors"]
                if len(ancestors) > 1:
                    ancestor_lists.append([m_gold_to_id[a] for a in ancestors[1:]])
                else:
                    ancestor_lists.append([])
            patterns = [[sht_builder.tree["nodes"][a]["info"]["cluster_id"] for a in ancestors] for ancestors in ancestor_lists]
            for pattern_id in range(len(patterns)):
                if pattern_id >= 1 and (not set(patterns[pattern_id]).issubset(set(patterns[pattern_id - 1]))):
                    strongly_templatized = False
                    break
            if strongly_templatized == False:
                break
            
        # well-formattedness
        sibling_same_cluster = True
        gold_m_parent_id_to_node_ids = {
            pid: [n["id"] for n in gold_nodes if n["ancestors"][-1] == pid]
            for pid in (list(m_gold_to_id.keys()) + [-1])
        }
        for pid in gold_m_parent_id_to_node_ids:
            siblings = gold_m_parent_id_to_node_ids[pid]
            if len(siblings) > 0:
                m_siblings = [m_gold_to_id[sid] for sid in siblings]
                assert [sht_builder.tree["nodes"][sid]["id"] for sid in m_siblings] == m_siblings
                if len(set([sht_builder.tree["nodes"][sid]["info"]["cluster_id"] for sid in m_siblings])) > 1:
                    sibling_same_cluster = False
                    break
        cluster_same_level = True
        m_cluster_id_to_node_ids = dict()
        for nid in list(m_gold_to_id.values()):
            cluster_id = sht_builder.tree["nodes"][nid]["info"]["cluster_id"]
            if cluster_id not in m_cluster_id_to_node_ids:
                m_cluster_id_to_node_ids[cluster_id] = []
            m_cluster_id_to_node_ids[cluster_id].append(nid)
        for cluster_id in m_cluster_id_to_node_ids:
            assert m_cluster_id_to_node_ids[cluster_id] == sorted(m_cluster_id_to_node_ids[cluster_id])
            ancestor_lists = []
            for nid in m_cluster_id_to_node_ids[cluster_id]:
                gold_nid = m_id_to_gold[nid]
                ancestors = gold_object_dicts_list[gold_nid]["ancestors"]
                print(cluster_id, gold_object_dicts_list[gold_nid]["text"], ancestors)#######
                if len(ancestor_lists) > 0 and len(ancestor_lists[-1]) != len(ancestors) - 1:
                    cluster_same_level = False
                    break
                else:
                    if len(ancestors) > 1:
                        ancestor_lists.append([m_gold_to_id[a] for a in ancestors[1:]])
                    else:
                        ancestor_lists.append([])
            if cluster_same_level == False:
                break

        

        m_file_to_stats[json_name] = {
            "is_correct_tree": is_correct_tree,
            "is_perfect_tree": is_perfect_tree,
            "is_sibling_same_cluster": sibling_same_cluster,
            "is_cluster_same_level": cluster_same_level,
            "is_well_formatted_tree": sibling_same_cluster and cluster_same_level,
            "is_strongly_templatized_tree": strongly_templatized,
            "num_correct_ancestor_nodes": num_correct_ancestor_nodes,
            "num_perfect_ancestor_nodes": num_perfect_ancestor_nodes,
            "num_correct_ts_nodes": num_correct_ts_nodes,
            "num_perfect_ts_nodes": num_perfect_ts_nodes,
            "num_nodes": len(nondummy_nodes),
            "prec_correct_ancestor_nodes": num_correct_ancestor_nodes / len(nondummy_nodes),
            "prec_perfect_ancestor_nodes": num_perfect_ancestor_nodes/ len(nondummy_nodes),
            "prec_correct_ts_nodes": num_correct_ts_nodes / len(nondummy_nodes),
            "prec_perfect_ts_nodes": num_perfect_ts_nodes / len(nondummy_nodes),
        }

        if strongly_templatized:
            assert is_correct_tree
        if sibling_same_cluster and cluster_same_level:
            assert strongly_templatized
            assert is_perfect_tree
        
        

    stats = {
        "num_correct_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_correct_tree"]]),
        "num_perfect_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_perfect_tree"]]),
        "num_well_formatted_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_well_formatted_tree"]]),
        "num_sibling_same_cluster_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_sibling_same_cluster"]]),
        "num_cluster_same_level_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_cluster_same_level"]]),
        "num_strongly_templatized_tree": len([f for f in m_file_to_stats if m_file_to_stats[f]["is_strongly_templatized_tree"]]),
        "avg_num_correct_ancestor_nodes": np.mean([v["prec_correct_ancestor_nodes"] for v in list(m_file_to_stats.values())]),
        "avg_num_perfect_ancestor_nodes": np.mean([v["prec_perfect_ancestor_nodes"] for v in list(m_file_to_stats.values())]),
        "avg_num_correct_ts_nodes": np.mean([v["prec_correct_ts_nodes"] for v in list(m_file_to_stats.values())]),
        "avg_num_perfect_ts_nodes": np.mean([v["prec_perfect_ts_nodes"] for v in list(m_file_to_stats.values())]),
        "details": m_file_to_stats,
    }


    with open("/mnt/f/Research-2024-spring/SHTRAG/comphrdoc_stats.json", 'w') as file:
        json.dump(stats, file, indent=4)

def test_sht_builder_correctness_using_gold_civic_sht():
    '''
    2024-07-24 18:18 - All 19 tests pass
    '''
    from sht import SHTBuilder, SHTBuilderConfig
    in_json_dir = "/mnt/f/Research-2024-spring/RetrieveSys/data/toc/json/gold/civic"
    out_dir = "/mnt/f/Research-2024-spring/SHTRAG/tmp"
    files = os.listdir(in_json_dir)
    for json_file in files:
        if not json_file.endswith(".json"):
            continue

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        
        with open(os.path.join(in_json_dir, json_name), 'r') as file:
            object_dicts_list = json.load(file)

        new_object_dicts_list = []

        for object_dict in object_dicts_list:
            new_object_dicts_list.append({
                "id": object_dict["hid"],
                "text": object_dict["text"],
                "type": "Title",
                "features": dict(),
                "cluster_id": len(object_dict["ancestors"]),
            })
        
        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(new_object_dicts_list)
        sht_builder.estimate_cost()
        sht_builder._add_nondummy_parent()
        sht_builder._add_nondummy_children()

        sht_builder.store2json()
        sht_builder.visualize()

        for node in sht_builder.tree["nodes"]:
            old_node = object_dicts_list[node["id"]]
            assert node["heading"] == old_node["text"]
            assert node["parent"] == old_node["ancestors"][-1]
            assert node["children"] == old_node["children"]
            assert node["nondummy_parent"] == node["parent"]
            assert node["nondummy_children"] == node["children"]

def test_sht_builder_with_load():
    from sht import SHTBuilder, SHTBuilderConfig
    pdf_dir = "/mnt/f/Research-2024-spring/RetrieveSys/data/pdf/civic"
    in_json_dir = "/mnt/f/Research-2024-spring/RetrieveSys/data/adobe_extract/civic"
    out_dir = "/mnt/f/Research-2024-spring/SHTRAG/database/civic/sht"
    files = os.listdir(in_json_dir)
    
    for json_file in files:
        if not json_file.endswith(".json"):
            continue

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        
        load_json = os.path.join("/mnt/f/Research-2024-spring/SHTRAG/database/civic/sht", json_name)
        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=load_json, chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(None)
        sht_builder.store2json()
        sht_builder.visualize()

def test_sht_builder():
    from sht import SHTBuilder, SHTBuilderConfig, ClusteringOracle, ClusteringOracleConfig
    pdf_dir = "/mnt/f/Research-2024-spring/financebench/pdfs"
    in_json_dir = "/mnt/f/Research-2024-spring/RetrieveSys/data/adobe_extract/finance"
    out_dir = "/mnt/f/Research-2024-spring/SHTRAG/database/finance/sht"
    files = os.listdir(in_json_dir)
    clustering_oracle_config = ClusteringOracleConfig(
        store_json=None,
    )
    clustering_oracle = ClusteringOracle(config=clustering_oracle_config)
    cluster_time_stats = []
    for json_file in files:
        if not json_file.endswith(".json"):
            continue

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        pdf_name = f"{file_name}.pdf"
        
        with open(os.path.join(in_json_dir, json_name), 'r') as file:
            object_dicts_list = json.load(file)
        new_object_dicts_list = clustering_oracle.cluster(pdf_path=os.path.join(pdf_dir, pdf_name), object_dicts_list=object_dicts_list)
        
        cluster_time = clustering_oracle.cluster_time
        cluster_time_stats.append(cluster_time)

        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=None, chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(new_object_dicts_list)
        sht_builder.store2json()
        sht_builder.visualize()

    print(f"cluster_time_stats={cluster_time_stats}\nmean_cluster_time={np.mean(cluster_time_stats)}\ntot_cluster_time={sum(cluster_time_stats)}")


def sht_builder_add_summary_with_load(start, end):
    from sht import SHTBuilder, SHTBuilderConfig
    dataset = "qasper"
    if dataset == "finance":
        raise ValueError("cannot add summary for finance set")
    in_json_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"
    out_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"
    files = [f for f in os.listdir(in_json_dir) if f.endswith(".json")]
    assert end <= len(files)
    add_summary_stats = dict()
    
    for json_file in files[start:end]:
        if not json_file.endswith(".json"):
            assert False

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        
        load_json = os.path.join(f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht", json_name)
        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=load_json, chunk_size=100, summary_len=100)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(None)
        stats = sht_builder.add_summaries()
        sht_builder.check()
        add_summary_stats[file_name] = stats
        sht_builder.store2json()
        sht_builder.visualize()
        
    tot_stats = dict()
    tot_input_tokens = sum([s["input_tokens"] for s in add_summary_stats.values()])
    tot_output_tokens = sum([s["output_tokens"] for s in add_summary_stats.values()])
    tot_time = sum([s["time"] for s in add_summary_stats.values()])
    tot_stats = {
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "tot_time": tot_time,
        "details": add_summary_stats,
    }

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/{dataset}[{start}:{end}]_sht_build_stats.json", 'w') as file:
        json.dump(tot_stats, file, indent=4)

def sht_builder_add_embeddings(embedding_model_name: str, dataset: str):
    from sht import SHTBuilder, SHTBuilderConfig
    if dataset == "finance":
        raise ValueError("cannot add embeddings for finance set")
    in_json_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"
    out_file_name_suffix = ""
    if embedding_model_name != "sbert":
        out_file_name_suffix = "-" + embedding_model_name
    out_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht{out_file_name_suffix}"
    files = [f for f in os.listdir(in_json_dir) if f.endswith(".json")]

    add_embeddings_stats = dict()
    
    for json_file in files:
        # print("=========================SHT================================")
        if not json_file.endswith(".json"):
            assert False

        file_name = json_file.replace(".json", "")
        json_name = f"{file_name}.json"
        
        load_json = os.path.join(in_json_dir, json_name)



        sht_builder_config = SHTBuilderConfig(store_json=os.path.join(out_dir, json_name), load_json=load_json, chunk_size=100, summary_len=100, embedding_model_name=embedding_model_name)
        sht_builder = SHTBuilder(sht_builder_config)
        sht_builder.build(None)

        node_ids_list = list(range(len(sht_builder.tree["nodes"])))

        stats = sht_builder.add_embeddings(node_ids_list)
        sht_builder.check()
        add_embeddings_stats[file_name] = stats
        sht_builder.store2json()
    tot_stats = dict()
    tot_hybrid_time = sum([s["hybrid"] for s in add_embeddings_stats.values()])
    tot_texts_time = sum([s["texts"] for s in add_embeddings_stats.values()])
    tot_heading_time = sum([s["heading"] for s in add_embeddings_stats.values()])
    tot_stats = {
        "tot_hybrid_time": tot_hybrid_time,
        "tot_texts_time": tot_texts_time,
        "tot_heading_time": tot_heading_time,
        "details": add_embeddings_stats
    }

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/{dataset}_sht{out_file_name_suffix}_embedding_stats.json", 'w') as file:
        json.dump(tot_stats, file, indent=4)


def sht_num_nodes(dataset):
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/sht"
    files = os.listdir(tree_dir)
    node_count = []
    for json_name in files:
        if not json_name.endswith(".json"):
            continue
        with open(os.path.join(tree_dir, json_name), 'r') as file:
            sht = json.load(file)
        node_count.append(len([n for n in sht["nodes"] if not n["is_dummy"]]))
        assert len([n for n in sht["nodes"] if "embeddings" in n]) == len([n for n in sht["nodes"] if not n["is_dummy"]])

    assert len(node_count) == len([f for f in files if f.endswith(".json")])
    
    print(len(node_count), np.mean(node_count))

if __name__ == "__main__":
    # start = 1
    # end = 416
    # sht_builder_add_summary_with_load(start, end)
    # sht_builder_add_embeddings(embedding_model_name="te3small", dataset="civic")
    sht_builder_add_embeddings(embedding_model_name="te3small", dataset="contract")
    sht_builder_add_embeddings(embedding_model_name="te3small", dataset="qasper")
    # test_summarization_model()
    # sht_num_nodes("qasper")