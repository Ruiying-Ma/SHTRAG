import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List
import json
import re
import config
import argparse
import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import logging
import logging_config
logging.disable(level=logging.DEBUG)



def get_eval_result(context_config, dataset, metric):
    if dataset == "civic1":
        return eval.eval_civic.civic_q1_eval_answer(context_config)
    elif dataset == "civic2":
        m_metric_index = {
            "recall": 0,
            "precision": 1,
            "f1": 2
        }
        return eval.eval_civic.civic_q2_eval_answer(context_config)[m_metric_index[metric]]
    elif dataset == "contract":
        return eval.eval_contract.contract_eval_answer(context_config)
    elif dataset == "qasper":
        if metric == "token":
            return eval.eval_qasper.qasper_eval_answer_f1(context_config)
        elif metric == "llmjudge":
            return eval.eval_qasper.qasper_eval_answer_llm(context_config)
        else:
            assert False, f"Metric {metric} not supported for Qasper."
        return eval.eval_qasper.qasper_eval_answer_llm(context_config)
    else:
        raise ValueError(f"Dataset {dataset} not supported yet.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["civic1", "civic2", "contract", "finance", "qasper"])
    parser.add_argument('--metric', type=str, required=False, choices=["recall", "precision", "f1", "token", "llmjudge"])
    args = parser.parse_args()

    if args.dataset in ["qasper"]: #TODO: add financebench
        check_answer = input(f"answer.jsonl has been extracted from qa_result.jsonl? [y/n]")
        assert check_answer.lower() == "y"
        check_rating = input(f"rating.jsonl has been extracted from rating_result.jsonl? [y/n]")
        assert check_rating.lower() == "y"
    
    if args.dataset in ["civic2", "qasper"]: # TODO: add financebench
        if args.dataset == "civic2":
            assert args.metric in ["recall", "precision", "f1"]
        if args.dataset == "qasper":
            assert args.metric in ["token", "llmjudge"]

    ##########################################End to end
    # tab_str = ""
    # for method in ["vanilla", "raptor", "sht"]:
    #     for context_len_ratio in config.CONTEXT_LEN_RATIO_LIST:
    #         for embedding_mode in config.NODE_EMBEDDING_MODEL_LIST:
    #             if method == "sht":
    #                 context_config = (method, None, embedding_mode, True, True, True, context_len_ratio)
    #             else:
    #                 context_config = (method, None, embedding_mode, None, None, None, context_len_ratio)
                
    #             if args.dataset in ["qasper"]: #TODO: add financebench
    #                 context_path = config.get_config_jsonl_path(args.dataset, context_config)
    #                 assert os.path.exists(context_path)
    #                 answer_path = context_path.replace("context.jsonl", "answer.jsonl")
    #                 assert os.path.exists(answer_path)
    #                 rating_path = context_path.replace("context.jsonl", "rating.jsonl")
    #                 assert os.path.exists(rating_path)
    #             print(context_config)
    #             result = get_eval_result(context_config, args.dataset, args.metric)
    #             tab_str += str(result) + "\t"

    #     tab_str += "\n"

    # print(tab_str)

    ############################################ ablation
    tab_str = ""
    for context_config in config.CONTEXT_CONFIG_LIST:
        logging.info(f"Evaluating context_config={context_config}...")
        result = get_eval_result(context_config, args.dataset, args.metric)
        tab_str += str(result) + "\n"

    print(tab_str)