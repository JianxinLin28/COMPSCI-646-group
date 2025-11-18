# Match hard passage to the original positive document (id)

# 1. Take in a hard-passage jsonl file (under negative_passages)
#    Make a dict with key=hard passage, value=pos doc

# 2. Give each hard passage a UUID

# 3. Find pos doc id for each pos doc from db

# 4. Make hp_id_to_hp

# 5. Make pos_id_to_hp_ids 

# 6. Load qrel from database

# 7. Make new qrel entries and save as file


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
hp_jsonl = "../outputs/negative_passages/IIYiClinical_50_train_data_2025-11-17_18-42-03.jsonl"
data_path = "../data/IIYiClinical"
qrel_hf_path = "hf://datasets/R2MED/IIYi-Clinical/qrels.jsonl"
output_path = "output"
os.makedirs(output_path, exist_ok=True)
output_data_path = os.path.join(output_path, "data")

hp_id_to_hp_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_hp_id_to_hp.jsonl"))
pos_id_to_hp_ids_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_pos_id_to_hp_ids.jsonl"))
qrel_extent_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_qrel_extent.jsonl"))
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


def get_pos_doc_id_to_hp_ids():
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
    return pos_doc_id_to_hp_ids


def load_qrel():
    hf_data_reader = BaseHFDataReader()
    hf_data_reader.hf_path = qrel_hf_path
    hf_data_reader.name = os.path.basename(data_path)
    hf_data_reader.save_dir = output_data_path
    hf_data_reader._load()
    
    q_ids = hf_data_reader.data_reader.column_to_list("train", "q_id")
    p_ids = hf_data_reader.data_reader.column_to_list("train", "p_id")
    return [(q_id, p_id) for q_id, p_id in zip(q_ids, p_ids)]


def get_qrel_extent(pos_doc_id_to_hp_ids, qrel):
    result = []

    for q_id, p_id in qrel:
        if p_id not in pos_doc_id_to_hp_ids:
            continue

        hp_ids = pos_doc_id_to_hp_ids[p_id]
        for hp_id in hp_ids:
            result.append({
                "q_id": q_id,
                "p_id": hp_id,
                "score": 1
            })
    return result


def main():
    pos_doc_id_to_hp_ids = get_pos_doc_id_to_hp_ids()
    qrel = load_qrel()
    qrel_extent = get_qrel_extent(pos_doc_id_to_hp_ids, qrel)

    # Save qrel_extent
    with open(qrel_extent_path, "w", encoding="utf-8") as f:
        for obj in qrel_extent:
            f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
