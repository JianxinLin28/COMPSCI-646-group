# Match hard passage to the original positive document (id)

# 1. Take in a hard-passage jsonl file (under negative_passages)
#    Make a dict with key=hard passage, value=pos doc

# 2. Give each hard passage a UUID

# 3. Find pos doc id for each pos doc from db

# 4. The final dict is key=hard passage id, value=pos doc id


import os
import sys
import json
from typing import List
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MyUtil/')))
import my_logger # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from hf_dataset_manager import HFDatasetManager
from data_reader import DataReader, BaseHFDataReader


my_logger= my_logger.MyLogger()


# ============================================
hp_jsonl = "../outputs/negative_passages/IIYiClinical_200_train_data_2025-11-17_16-41-06.jsonl"
data_path = "../data/IIYiClinical"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

hp_id_to_hp_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_hp_id_to_hp.jsonl"))
pos_id_to_hp_ids_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_pos_id_to_hp_ids.jsonl"))
# ============================================


def get_hp_to_pos_doc():
    result = dict()
    with open(hp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pos = item["pos"][0][1]
            hp = item["neg"][0][1]
            result[hp] = pos
    return result


def get_hp_to_uuid(hps: List[str]) -> dict[str, str]:
    return {hp : str(uuid.uuid5(uuid.NAMESPACE_DNS, hp)) for hp in hps}


def get_id_for_pos_doc():
    result = dict()
    hf_manager = HFDatasetManager(data_path)
    hf_manager.load()
    data_reader = DataReader(hf_manager)
    hf_data_reader = BaseHFDataReader()
    hf_data_reader.data_reader = data_reader
    paragraphs, pids = hf_data_reader.get_documents()
    for pid, paragraph in zip(pids, paragraphs):
        result[paragraph] = pid
    return result


def get_hp_id_to_pos_doc_id():
    id_for_pos_doc = get_id_for_pos_doc()

    hp_to_pos_doc = get_hp_to_pos_doc()
    hps = list(hp_to_pos_doc.keys())
    hp_to_uuid = get_hp_to_uuid(hps)

    output1 = []
    for hp, uuid in hp_to_uuid.items():
        output1.append({
            "hard_passage_id": uuid,
            "hard_passage": hp
        })

    # Save hp id to hp
    with open(hp_id_to_hp_path, "w", encoding="utf-8") as f:
        for obj in output1:
            f.write(json.dumps(obj) + "\n")

    result = dict()
    for hp, pos_doc in hp_to_pos_doc.items():
        result[hp_to_uuid[hp]] = id_for_pos_doc[pos_doc]
    return result


def main():
    hp_id_to_pos_doc_id = get_hp_id_to_pos_doc_id()
    
    # Save to output, key=pos doc id, value=[hp_id]
    pos_doc_id_to_hp_ids = dict()
    for hp_id, pos_doc_id in hp_id_to_pos_doc_id.items():
        if pos_doc_id not in pos_doc_id_to_hp_ids:
            pos_doc_id_to_hp_ids[pos_doc_id] = [hp_id]
        else:
            pos_doc_id_to_hp_ids[pos_doc_id].append(hp_id)

    output2 = []
    for pos_doc_id, hp_ids in pos_doc_id_to_hp_ids.items():
        output2.append({
            "pos_id": pos_doc_id,
            "hard_passage_ids": hp_ids
        })

    # Save pos id to hp ids
    with open(pos_id_to_hp_ids_path, "w", encoding="utf-8") as f:
        for obj in output2:
            f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
