import json


def calc_sht_stats():
    start = 268
    end = 416
    with open("/mnt/f/Research-2024-spring/SHTRAG/qasper[1:416]_sht_build_stats.json", 'r') as file:
        sht_build_data = json.load(file)

    with open("/mnt/f/Research-2024-spring/SHTRAG/qasper[1:416]_sht_embedding_stats.json", 'r') as file:
        sht_embedding_data = json.load(file)

    with open("/mnt/f/Research-2024-spring/SHTRAG/qasper[268:416]_raptor_build_and_embedding_stats.json", 'r') as file:
        raptor_data = json.load(file)

    # check raptor
    print(sum([o["hybrid"] for o in sht_embedding_data["details"].values()]))
    print(sum([o["texts"] for o in sht_embedding_data["details"].values()]))
    print(sum([o["heading"] for o in sht_embedding_data["details"].values()]))
    # print(sum([o["embedding_time"] for o in sht_build_data["details"].values()]))

    new_sht_build_data = {
        "tot_input_tokens": 0,
        "tot_output_tokens": 0,
        "tot_time": 0.0,
        "details": dict(),
    }

    new_sht_embedding_data = {
        "tot_hybrid_time": 0.0,
        "tot_texts_time": 0.0,
        "tot_heading_time": 0.0,
        "details": dict(),
    }

    file_names = list(sht_build_data["details"].keys())
    assert sorted(file_names) == file_names
    print(file_names[-1])
    print(len(file_names))
    print(len(file_names[(start - 1):(end - 1)]))

    for file_name in file_names[(start - 1):(end - 1)]:
        new_sht_build_data["tot_input_tokens"] += sht_build_data["details"][file_name]["input_tokens"]
        new_sht_build_data["tot_output_tokens"] += sht_build_data["details"][file_name]["output_tokens"]
        new_sht_build_data["tot_time"] += sht_build_data["details"][file_name]["time"]
        new_sht_build_data["details"][file_name] = sht_build_data["details"][file_name]

        new_sht_embedding_data["tot_hybrid_time"] += sht_embedding_data["details"][file_name]["hybrid"]
        new_sht_embedding_data["tot_texts_time"] += sht_embedding_data["details"][file_name]["texts"]
        new_sht_embedding_data["tot_heading_time"] += sht_embedding_data["details"][file_name]["heading"]
        new_sht_embedding_data["details"][file_name] = sht_embedding_data["details"][file_name]

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/qasper[{start}:{end}]_sht_build_stats.json", 'w') as file:
        json.dump(new_sht_build_data, file, indent=4)

    with open(f"/mnt/f/Research-2024-spring/SHTRAG/qasper[{start}:{end}]_sht_embedding_stats.json", 'w') as file:
        json.dump(new_sht_embedding_data, file, indent=4)

if __name__ == "__main__":
    calc_sht_stats()