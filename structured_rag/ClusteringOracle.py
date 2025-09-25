import json
import os
from typing import Dict, List
import logging
import time

from .FeatureExtractor import FeatureExtractor


class ClusteringOracleConfig:
    def __init__(self, store_json, thresh_is_centered=2.0, round_font_size=2, thresh_is_underlined=6.0, thresh_is_line=2.0, thresh_rect=0.0):
        '''
        Args:
            - store_json: if not stored, set `None`; else, set as the absolute output path
        '''

        if store_json is not None:
            if not store_json.endswith(".json"):
                raise ValueError("store_json must be a path to a json file that ends with .json")
            elif not os.path.exists(os.path.dirname(store_json)):
                raise ValueError("store_json directory doesn't exist")
        
        if not isinstance(thresh_is_centered, float):
            raise ValueError("thresh_is_centered should be a float number")

        if not isinstance(round_font_size, int):
            raise ValueError("round_font_size should be an integer")

        if not isinstance(thresh_is_underlined, float):
            raise ValueError("thresh_is_underlined should be a float number")
        
        if not isinstance(thresh_is_line, float):
            raise ValueError("thresh_is_line should be a float number")

        if not isinstance(thresh_rect, float):
            raise ValueError("thresh_rect should be a float number")
        
        self.store_json = store_json
        self.thresh_is_centered = thresh_is_centered
        self.round_font_size = round_font_size
        self.thresh_is_underlined = thresh_is_underlined
        self.thresh_is_line = thresh_is_line
        self.thresh_rect = thresh_rect

class ClusteringOracle:
    '''
    Take a PDF's its headings & list items as SHT nodes. Cluster the nodes according to their visual patterns.
    '''
    def __init__(self, config: ClusteringOracleConfig):
        if not isinstance(config, ClusteringOracleConfig):
            raise ValueError("config must be an instance of ClusteringOracleConfig")

        self.store_json = config.store_json
        
        self.feature_extractor = FeatureExtractor(
            thresh_is_centered=config.thresh_is_centered,
            round_font_size=config.round_font_size,
            thresh_is_underlined=config.thresh_is_underlined,
            thresh_is_line=config.thresh_is_line,
            thresh_rect=config.thresh_rect,
        )

        self.cluster_time = None

    def cluster(self, pdf_path: str, object_dicts_list: List[Dict]) -> List[Dict]:
        '''
        Extract an object's visual pattern. Add them to `object["features"]`.

        Args:
            - pdf_path (str)
            - object_dicts_list (List[Dict]): VGT's classification. An `object` in `object_dicts_list` is a bounding box. It stores:
            ```
            {
                "left": float, # bounding box
                "top": float, # bounding box
                "width": float, # bounding box
                "height": float, # bounding box
                "page_number": int, # the page where it locates. 1-based.
                "page_width": int,
                "page_height": int,
                "text": str, # the text inside the bounding box; cleaned for Section headers/Titles
                "type": str, # VGT classification
            }
            ```

        Returns:
            - new_object_dicts_list (List[Dict]): Cluster the Section headers/Titles/List items. The newly added keys are:
            ```
            {
                "features": dict,
                "cluster_id": int, # only for Section headers/Titles/List items
            }
            ```
        '''

        self.cluster_time = None
        start_time = time.time()
        
        self.feature_extractor.load_pdf_doc(pdf_path)
        
        if not isinstance(object_dicts_list, List):
            raise ValueError("objects must be a list")
        else:
            canonical_layout = {
                "left": float,
                "top": float,
                "width": float,
                "height": float,
                "page_number": int,
                "page_width": int,
                "page_height": int,
                "text": str,
                "type": str,
            }
            for object_dict in object_dicts_list:
                if not isinstance(object_dict, Dict):
                    raise ValueError("object must be a dictionary")
                # Check if all expected keys are in the dictionary and have the correct type
                for key, value_type in canonical_layout.items():
                    if key not in object_dict:
                        raise ValueError(f"Omitted key of object: {key}")
                    elif not isinstance(object_dict[key], value_type):
                        raise ValueError(f"Wrong value type of object: key '{key}' should be of type {value_type}, but is of type {type(object_dict[key])}")
                # Check if there are no extra keys in the dictionary
                for key in object_dict.keys():
                    if key not in canonical_layout:
                        raise ValueError(f"Unknown key of object: {key}")
                # Check if page_number is 1-based
                if object_dict["page_number"] < 1:
                    raise ValueError(f"Wrong value of page_number: {object_dict['page_number']}. Must be 1-based.")
                    
        new_object_dicts_list = []

        prev_header_object_id = -1

        object_id = 0

        for object_dict in object_dicts_list:
            page_number = object_dict["page_number"]
            object_type = object_dict["type"]
            if object_type == "Section header" or object_type == "Title":
                self.feature_extractor.load_page_and_underlines(page_number)
                self.feature_extractor.load_rect(
                    left=object_dict["left"],
                    top=object_dict["top"],
                    width=object_dict["width"],
                    height=object_dict["height"],
                    page_number=page_number,
                )

                object_dict["text"] = self.feature_extractor.extract_clean_text()

                if object_dict["text"].strip() == "":
                    continue

                feature_info = self.feature_extractor.extract_font_info()
                feature_info["is_centered"] = self.feature_extractor.is_centered()
                feature_info["is_underlined"] = self.feature_extractor.is_underlined()
                feature_info["is_all_cap"] = self.feature_extractor.is_all_cap(object_dict["text"])

                object_dict['id'] = object_id
                object_dict["features"] = feature_info
                prev_header_object_id = object_id
                object_id += 1
                new_object_dicts_list.append(object_dict)
            else:
                if object_dict["text"].strip() == "":
                    continue

                if object_type == "List item":
                    feature_info = {
                        "parent_id": prev_header_object_id,
                    }

                    object_dict["id"] = object_id
                    object_dict["features"] = feature_info
                    object_id += 1
                    new_object_dicts_list.append(object_dict)
                elif object_type != "Picture":
                    assert object_type in ["Caption", "Footnote", "Formula", "Page footer", "Page header", "Table", "Text"]
                    object_dict["id"] = object_id
                    object_id += 1
                    new_object_dicts_list.append(object_dict)
                else:
                    assert object_type == "Picture"
                    continue

        # check new_object_dicts_list object_id
        assert list([o["id"] for o in new_object_dicts_list]) == list(range(len(new_object_dicts_list)))
        assert all([o["text"] != "" for o in new_object_dicts_list])
        
        # extract list types
        candid_object_dicts_list = [o for o in new_object_dicts_list if o["type"] in ["Section header", "Title", "List item"]]
        texts_list = [o["text"] for o in candid_object_dicts_list]
        list_types_list = self.feature_extractor.extract_list_type(texts_list)
        for object_dict, list_type in zip(candid_object_dicts_list, list_types_list):
            assert "features" in object_dict
            assert "list_type" not in object_dict["features"]
            object_dict["features"]["list"] = list_type

        # cluster
        m_feature_strings_to_cluster_id = dict()
        for object_dict in candid_object_dicts_list:
            feature_string = json.dumps(tuple(sorted(object_dict["features"].items())))
            if feature_string not in m_feature_strings_to_cluster_id:
                new_cluster_id = len(m_feature_strings_to_cluster_id)
                m_feature_strings_to_cluster_id[feature_string] = new_cluster_id
            object_dict["cluster_id"] = m_feature_strings_to_cluster_id[feature_string]

        # timing
        end_time = time.time()
        self.cluster_time = end_time - start_time

        # store
        self.store2json(new_object_dicts_list)

        # reset feature_extractor
        self.feature_extractor.reset()


        return new_object_dicts_list


    def store2json(self, object_dicts_list):
        if self.store_json:
            with open(self.store_json, 'w') as file:
                json.dump(object_dicts_list, file, indent=4)
            logging.info(f"Successfully stored to {self.store_json}")


        





        
        