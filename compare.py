from structured_rag import SHTBuilderConfig, SHTBuilder
from structured_rag.ClusteringOracle import ClusteringOracle, ClusteringOracleConfig
import os
import fitz
from difflib import SequenceMatcher
import json
from structured_rag.utils import get_nondummy_ancestors
import logging
logging.disable(logging.INFO)
import sys
import traceback

def get_coord(bb, direction, need_rescale, err=2.0):
    if isinstance(bb, fitz.Rect):
        if direction == "left":
            if not need_rescale:
                return float(max(bb.x0 - err, 0))
            else:
                return 0.0
        elif direction == "top":
            return float(max(bb.y0 - err, 0))
        elif direction == "width":
            return float(bb.x1 - bb.x0 + err * 2)
        elif direction == "height":
            return float(bb.y1 - bb.y0 + err * 2)
        else:
            raise ValueError(f"Unknown direction {direction}")
    elif isinstance(bb, fitz.Quad):
        return get_coord(bb.rect, direction)
    else:
        raise ValueError(f"Unknown instance {bb}")

def find_bounding_boxes(pdf, text, vgt_identified_headings, need_rescale):
    assert isinstance(pdf, fitz.Document)
    bounding_boxes = dict()
    for vgt_heading in vgt_identified_headings:
        if text in vgt_heading["text"]:
            page_id = vgt_heading["page_number"]
            if page_id not in bounding_boxes:
                bounding_boxes[page_id] = []
            bounding_boxes[page_id].append(vgt_heading)
    if len(bounding_boxes) == 0:
        for page_id in range(len(pdf)):
            page = pdf[page_id]
            candid_bbs = page.search_for(text)
            if len(candid_bbs) > 0:
                if not page_id + 1 in bounding_boxes:
                    bounding_boxes[page_id + 1] = []
                for candid_bb in candid_bbs:
                    bounding_boxes[page_id + 1].append({
                        "left": get_coord(candid_bb, "left", need_rescale),
                        "top": get_coord(candid_bb, "top", need_rescale),
                        "width": get_coord(candid_bb, "width", need_rescale),
                        "height": get_coord(candid_bb, "height", need_rescale),
                        "page_number": page_id + 1,
                        "page_width": int(page.rect.width),
                        "page_height": int(page.rect.height),
                        "text": text,
                        "type": "Section header"
                    })

    if len(bounding_boxes) == 0 and len(text) > 2:
        bounding_boxes.update(find_bounding_boxes(pdf, text[:int(len(text)/2)], vgt_identified_headings, need_rescale))

    if len(bounding_boxes) == 0 and len(text) > 2:
        bounding_boxes.update(find_bounding_boxes(pdf, text[-int(len(text)/2):], vgt_identified_headings, need_rescale))

    assert len(bounding_boxes) > 0
    assert all(pid >= 1 for pid in bounding_boxes.keys())
    
    return bounding_boxes


    
def create_headings(pdf_path, need_rescale):
    name = os.path.basename(pdf_path).replace(".pdf", "")
    heading_identification_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "intrinsic"), "heading_identification", name+".json")
    os.makedirs(os.path.dirname(heading_identification_path), exist_ok=True)

    vgt_identified_heading_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "heading_identification"),  name+".json")
    with open(vgt_identified_heading_path, 'r') as file:
        vgt_result = json.load(file)
        vgt_identified_headings = [h for h in vgt_result if h["type"] in ["Title", "Section header", "List item"]]

    if not os.path.exists(heading_identification_path):
        pdf = fitz.open(pdf_path)
        intrinsic_sht_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "grobid"), "sbert.gpt-4o-mini.c100.s100", "sht", name+".json") ############################################################
        with open(intrinsic_sht_path, 'r') as file:
            intrinsic_sht = json.load(file)["nodes"]
        candid_bounding_boxes = []
        for node in intrinsic_sht:
            if node["type"] == "text":
                continue
            assert node["heading"].strip() != ""
            bb_dict = find_bounding_boxes(pdf, node["heading"], vgt_identified_headings, need_rescale)
            assert len(bb_dict) > 0
            candid_bounding_boxes.append(bb_dict)
        
        positions = [] # page_id, left, top, width, height
        cur_page = 1
        cur_top = 0.0
        for candid_bb_dict in candid_bounding_boxes:
            min_page_id = sys.maxsize
            min_top = sys.maxsize
            min_bb = None
            for page_id, candid_bb_list in candid_bb_dict.items():
                if page_id < cur_page:
                    continue
                for candid_bb in candid_bb_list:
                    if page_id == cur_page and candid_bb["top"] <= cur_top:
                        continue
                    if page_id < min_page_id:
                        min_page_id = page_id
                        min_top = candid_bb["top"]
                        min_bb = candid_bb
                    if page_id == min_page_id and candid_bb["top"] < min_top:
                        min_top = candid_bb["top"]
                        min_bb = candid_bb
            assert min_bb != None
            cur_page = min_bb["page_number"]
            cur_top = min_bb["top"]
            positions.append(min_bb)

        
        with open(heading_identification_path, 'w') as file:
            json.dump(positions, file, indent=4)

    with open(heading_identification_path, 'r') as file:
        headings = json.load(file)
    return headings

def load_headings(pdf_path):
    heading_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "intrinsic"), "node_clustering", os.path.basename(pdf_path).replace(".pdf", ".json"))
    if os.path.exists(heading_path):
        with open(heading_path, 'r') as file:
            raw_headings = json.load(file)
        headings = []
        for h in raw_headings:
            for key_name in ["id", "features", "cluster_id"]:
                if key_name in h:
                    del h[key_name]
            assert set(h.keys()) == {"left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"}
            headings.append(h)
    else:
        headings = create_headings(pdf_path, need_rescale=False)
    return headings

def build_sht_skeleton(pdf_path):
    headings = load_headings(pdf_path)

    clustering_oracle = ClusteringOracle(ClusteringOracleConfig(store_json=None))

    sht_builder = SHTBuilder(SHTBuilderConfig(
        store_json=os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_temp_output.json"),
        load_json=None,
        chunk_size=100,
        summary_len=100,
        embedding_model_name="sbert",
        summarization_model_name="empty",
    ))
    # build SHT skeleton
    sht_builder.build(clustering_oracle.cluster(
        pdf_path=pdf_path,
        object_dicts_list=headings
    ))

    if os.path.basename(pdf_path) == "01262022-1835.pdf":
        print(f"store {os.path.basename(pdf_path)} to json/vis")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_temp_output.json"), 'w') as file:
            json.dump(sht_builder.tree, file, indent=4)
        sht_builder.visualize(vis_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_temp_output.vis"))
    
    return sht_builder.tree["nodes"]

def tree_is_satisfied(raw_sht, condition, pdf_path):
    
    # load SHT
    sht = [n for n in raw_sht if (n["is_dummy"] == False) and (n["type"] != "text")]
    sht_node_ids = [n["id"] for n in sht]

    # load intrinsic SHT
    name = os.path.basename(pdf_path).replace(".pdf", "")
    intrinsic_sht_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "intrinsic"), "sbert.gpt-4o-mini.c100.s100", "sht", name+".json")
    with open(intrinsic_sht_path, 'r') as file:
        raw_intrinsic_sht = json.load(file)["nodes"]
        intrinsic_sht = [n for n in raw_intrinsic_sht if n["type"] != "text"]
    
    if len(sht) != len(intrinsic_sht):
        return False

    m_intrinsic_id_sht_id = dict()
    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        m_intrinsic_id_sht_id[intrinsic_sht_node["id"]] = sht_node["id"]
    

    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        assert m_intrinsic_id_sht_id[intrinsic_sht_node["id"]] == sht_node["id"]
        raw_sht_ancestors = get_nondummy_ancestors(raw_sht, sht_node["id"])
        sht_ancestors = [a for a in raw_sht_ancestors if a in sht_node_ids]
        intrinsic_ancestors = get_nondummy_ancestors(raw_intrinsic_sht, intrinsic_sht_node["id"])
        converted_intrinsic_ancestors = [m_intrinsic_id_sht_id[i] for i in intrinsic_ancestors if i in m_intrinsic_id_sht_id]
        # c-correct
        if condition == "relax":
            if not (set(converted_intrinsic_ancestors)).issubset(set(sht_ancestors)):
                return False
        # intrinsic
        else:
            assert condition == "strict"
            if not (set(sht_ancestors))==(set(converted_intrinsic_ancestors)):
                return False        
    return True

def tree_count_satisfied(raw_sht, condition, pdf_path):
    
    # load SHT
    sht = [n for n in raw_sht if (n["is_dummy"] == False) and (n["type"] != "text")]
    sht_node_ids = [n["id"] for n in sht]

    # load intrinsic SHT
    name = os.path.basename(pdf_path).replace(".pdf", "")
    intrinsic_sht_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "intrinsic"), "sbert.gpt-4o-mini.c100.s100", "sht", name+".json")
    with open(intrinsic_sht_path, 'r') as file:
        raw_intrinsic_sht = json.load(file)["nodes"]
        intrinsic_sht = [n for n in raw_intrinsic_sht if n["type"] != "text"]
    
    if len(sht) != len(intrinsic_sht):
        return None

    m_intrinsic_id_sht_id = dict()
    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        m_intrinsic_id_sht_id[intrinsic_sht_node["id"]] = sht_node["id"]
    
    cnt = 0
    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        assert m_intrinsic_id_sht_id[intrinsic_sht_node["id"]] == sht_node["id"]
        raw_sht_ancestors = get_nondummy_ancestors(raw_sht, sht_node["id"])
        sht_ancestors = [a for a in raw_sht_ancestors if a in sht_node_ids]
        intrinsic_ancestors = get_nondummy_ancestors(raw_intrinsic_sht, intrinsic_sht_node["id"])
        converted_intrinsic_ancestors = [m_intrinsic_id_sht_id[i] for i in intrinsic_ancestors if i in m_intrinsic_id_sht_id]
        if condition == "relax":
            if (set(converted_intrinsic_ancestors)).issubset(set(sht_ancestors)):
                cnt += 1
        else:
            assert condition == "strict"
            if (set(sht_ancestors))==(set(converted_intrinsic_ancestors)):
                cnt += 1   
    if len(sht) == 0:
        return None
    return cnt / len(sht)     





def doc_is_satisfied(raw_sht, condition, pdf_path):
    def get_cluster_id(m_intrinsic_id_sht_id, intrinsic_id, m_sht_id_sht):
        return m_sht_id_sht[m_intrinsic_id_sht_id[intrinsic_id]]["info"]["cluster_id"]
    
    # load SHT
    sht = [n for n in raw_sht if (n["is_dummy"] == False) and (n["type"] != "text")]
    assert all(["cluster_id" in n["info"] for n in sht])
    m_sht_id_sht = {
        n["id"]: n
        for n in sht
    }

    # load intrinsic SHT
    name = os.path.basename(pdf_path).replace(".pdf", "")
    intrinsic_sht_path = os.path.join(os.path.dirname(pdf_path).replace("pdf", "intrinsic"), "sbert.gpt-4o-mini.c100.s100", "sht", name+".json")
    with open(intrinsic_sht_path, 'r') as file:
        raw_intrinsic_sht = json.load(file)["nodes"]
        intrinsic_sht = [n for n in raw_intrinsic_sht if n["type"] != "text"]
    
    assert len(sht) == len(intrinsic_sht)

    m_intrinsic_id_sht_id = dict()
    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        m_intrinsic_id_sht_id[intrinsic_sht_node["id"]] = sht_node["id"]
    
    m_intrinsic_id_ancestor_clusters = dict()
    m_intrinsic_id_ancestors = dict()
    for sht_node, intrinsic_sht_node in zip(sht, intrinsic_sht):
        assert sht_node["id"] == m_intrinsic_id_sht_id[intrinsic_sht_node["id"]]
        intrinsic_ancestors = get_nondummy_ancestors(raw_intrinsic_sht, intrinsic_sht_node["id"])
        m_intrinsic_id_ancestors[intrinsic_sht_node["id"]] = [intrinsic_id for intrinsic_id in intrinsic_ancestors if intrinsic_id in m_intrinsic_id_sht_id]
        m_intrinsic_id_ancestor_clusters[intrinsic_sht_node["id"]] = [get_cluster_id(m_intrinsic_id_sht_id, intrinsic_id, m_sht_id_sht) for intrinsic_id in intrinsic_ancestors if intrinsic_id in m_intrinsic_id_sht_id] + [sht_node["info"]["cluster_id"]]

    for intrinsic_sht_id, cluster_list in m_intrinsic_id_ancestor_clusters.items():
        assert len(cluster_list) > 0
        for another_intrinsic_sht_id, another_cluster_list in m_intrinsic_id_ancestor_clusters.items():
            if intrinsic_sht_id <= another_intrinsic_sht_id:
                continue
            assert len(another_cluster_list) > 0
            # c-templatization
            if condition == "c-templatization":
                if another_cluster_list[-1] == cluster_list[-1]:
                    if not set(cluster_list[:-1]).issubset(set(another_cluster_list[:-1])):
                        return False
            # well-formatted constraint 1: siblings
            elif condition == "well-formattedness-1":
                ancestor_list = m_intrinsic_id_ancestors[intrinsic_sht_id]
                another_ancestor_list = m_intrinsic_id_ancestors[another_intrinsic_sht_id]
                parent_id = -1 if len(ancestor_list) == 0 else ancestor_list[-1]
                another_parent_id = -1 if len(another_ancestor_list) == 0 else another_ancestor_list[-1]
                if parent_id == another_parent_id:
                    if cluster_list[-1] != another_cluster_list[-1]:
                        return False
            # well-formatted constraint 2: cluster list
            elif condition == "well-formattedness-2":
                if another_cluster_list[-1] == cluster_list[-1]:
                    if cluster_list != another_cluster_list:
                        return False
            else:
                raise ValueError(f"Unknown condition {condition}")

    return True

def calc(dataset):
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "pdf")

    n_c_templatization = 0
    n_well_formattedness = 0
    n_well_formattedness_1 = 0
    n_well_formattedness_2 = 0
    n_c_correctness = 0
    n_intrinsic = 0
    
    for pdf_name in os.listdir(pdf_dir):
        
        print(pdf_name)
        assert pdf_name.endswith(".pdf")
        pdf_path = os.path.join(pdf_dir, pdf_name)
        raw_sht = build_sht_skeleton(pdf_path)
        is_c_correct = tree_is_satisfied(raw_sht, "relax", pdf_path)
        is_intrinsic = tree_is_satisfied(raw_sht, "strict", pdf_path)
        is_c_templatized = doc_is_satisfied(raw_sht, "c-templatization", pdf_path)
        is_well_formatted_1 = doc_is_satisfied(raw_sht, "well-formattedness-1", pdf_path)
        is_well_formatted_2 = doc_is_satisfied(raw_sht, "well-formattedness-2", pdf_path)
        is_well_formatted = (is_well_formatted_1 and is_well_formatted_2)
        
        if is_c_templatized:
            assert is_c_correct
        if is_well_formatted:
            assert is_intrinsic
        if is_intrinsic:
            assert is_c_correct

        print(f"\tc-templatization: {is_c_templatized}")
        print(f"\twell-formatted: {is_well_formatted}")   
        print(f"\t\t1: {is_well_formatted_1}")    
        print(f"\t\t2: {is_well_formatted_2}")    
        print(f"\tc-correctness: {is_c_correct}")
        print(f"\tintrinsic: {is_intrinsic}")

        if is_c_templatized:
            n_c_templatization += 1
        if is_well_formatted:
            n_well_formattedness += 1
        if is_well_formatted_1:
            n_well_formattedness_1 += 1
        if is_well_formatted_2:
            n_well_formattedness_2 += 1
        if is_c_correct:
            n_c_correctness += 1
        if is_intrinsic:
            n_intrinsic += 1

    n_files = len(os.listdir(pdf_dir))

    print(f"{dataset}: among {n_files} files\n\t%c_templatization = {round(n_c_templatization * 100 / n_files, 3)}\n\t%well_formattedness = {round(n_well_formattedness * 100 / n_files, 3)}\n\t\t1: {round(n_well_formattedness_1 * 100 / n_files, 3)}\n\t\t2: {round(n_well_formattedness_2 * 100 / n_files, 3)}\n\t%c_correctness = {round(n_c_correctness * 100 / n_files, 3)}\n\t%intrinsic = {round(n_intrinsic * 100 / n_files, 3)}")

def count(dataset):
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "pdf")

    tot_n_hrobust = 0.0
    tot_n_hintrinsic = 0.0
    n_files = 0
    
    for pdf_name in os.listdir(pdf_dir):
        
        print(pdf_name)
        assert pdf_name.endswith(".pdf")
        pdf_path = os.path.join(pdf_dir, pdf_name)
        raw_sht = build_sht_skeleton(pdf_path)
        n_hrobust = tree_count_satisfied(raw_sht, "relax", pdf_path)
        n_hintrinsic = tree_count_satisfied(raw_sht, "strict", pdf_path)

        if n_hrobust == None or n_hintrinsic == None:
            continue
        
        print(f"\t%hierarchy-robust nodes: {n_hrobust}")
        print(f"\t%hierarchy-intrinsic nodes: {n_hintrinsic}")   

        n_files += 1
        tot_n_hintrinsic += n_hintrinsic
        tot_n_hrobust += n_hrobust

    print(f"{dataset}: among {n_files} files\n\tAVG(%hierarchy-robust nodes) = {round(tot_n_hrobust * 100 / n_files, 3)}\n\tAVG(%hierarchy-intrinsic nodes) = {round(tot_n_hintrinsic * 100 / n_files, 3)}")

if __name__ == "__main__": 
    count("qasper")