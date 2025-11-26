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
from pathlib import Path
import sys
import json
import argparse
from typing import List, Tuple
import uuid


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil/')))
import my_logger # type: ignore

from data_reader import MedicalSciencesDataReader, PMCTreatmentDataReader, IIYiClinicalDataReader


my_logger= my_logger.MyLogger()


# ============================================
hp_jsonl_folder = "./outputs/negative_passages"
output_path = "outputs/qrel"
os.makedirs(output_path, exist_ok=True)
# ============================================


def get_jsonl_files(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.jsonl') and os.path.isfile(os.path.join(dir_path, f))
    ]


def get_hp_to_pos_doc(hp_jsonl):
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


def get_id_for_pos_doc(documents, doc_ids):
    result = dict()
    for pid, paragraph in zip(doc_ids, documents):
        result[paragraph] = pid
    return result


def get_hp_id_to_pos_doc_id(documents, doc_ids, hp_id_to_hp_path, hp_jsonl):
    id_for_pos_doc = get_id_for_pos_doc(documents, doc_ids)

    hp_to_pos_doc = get_hp_to_pos_doc(hp_jsonl)
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


def get_pos_doc_id_to_hp_ids(documents, doc_ids, pos_id_to_hp_ids_path, hp_id_to_hp_path, hp_jsonl):
    hp_id_to_pos_doc_id = get_hp_id_to_pos_doc_id(documents, doc_ids, hp_id_to_hp_path, hp_jsonl)
    
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


def load_qrel(qids, pids):
    return [(q_id, p_id) for q_id, p_id in zip(qids, pids)]


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


def main(hp_jsonl, documents, doc_ids, qids, pids):
    hp_id_to_hp_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_hp_id_to_hp.jsonl"))
    pos_id_to_hp_ids_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_pos_id_to_hp_ids.jsonl"))
    qrel_extent_path = os.path.join(output_path, os.path.basename(hp_jsonl).replace(".jsonl", "_qrel_extent.jsonl"))

    pos_doc_id_to_hp_ids = get_pos_doc_id_to_hp_ids(documents, doc_ids, pos_id_to_hp_ids_path, hp_id_to_hp_path, hp_jsonl)
    qrel = load_qrel(qids, pids)
    qrel_extent = get_qrel_extent(pos_doc_id_to_hp_ids, qrel)

    # Save qrel_extent
    with open(qrel_extent_path, "w", encoding="utf-8") as f:
        for obj in qrel_extent:
            f.write(json.dumps(obj) + "\n")


def get_generation_record(dataset: str) -> List[dict[str, str]]:
    result = []
    folder = Path("outputs/generation_record")
    files = [p for p in folder.iterdir() if p.name.startswith(dataset)]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    content = json.loads(line)
                    result.append(content)
    return result


def get_qids_pids(dataset: str) -> Tuple[List[str], List[str]]:
    qids = []
    pids = []
    generation_record = get_generation_record(dataset)
    for pair in generation_record:
        qids.append(pair["q_id"])
        pids.append(pair["p_id"])
    return qids, pids


def get_pids(qid_to_pids: dict[str, List[str]]) -> List[str]:
    result = []
    for qid, pids in qid_to_pids.items():
        result.extend(pids)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MedicalSciences', help='the dataset, see README')
    args = parser.parse_args()
    
    documents: List[str] = None
    doc_ids: List[str] = None

    match args.dataset:
        case "MedicalSciences":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = MedicalSciencesDataReader().get_documents()
        case "PMCTreatment":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = PMCTreatmentDataReader().get_documents()
        case "IIYiClinical":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = IIYiClinicalDataReader().get_documents()
        case _:
            my_logger.error("Invalid dataset. Please see README for valid datasets.")
    

    qids, pids = get_qids_pids(args.dataset)

    jsonl_files = get_jsonl_files(hp_jsonl_folder)
    for jsonl_file in jsonl_files:
        main(jsonl_file, documents, doc_ids, qids, pids)
