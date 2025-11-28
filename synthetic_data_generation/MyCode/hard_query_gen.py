import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime

from bm25_miner import BM25_Miner
from data_reader import (MedicalSciencesDataReader, PMCTreatmentDataReader, IIYiClinicalDataReader, 
                            MedicalSciencesQrelDataReader, PMCTreatmentQrelDataReader, IIYiClinicalQrelDataReader)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../synthetic_data_generation/')))
# from hard_negative_mining import BM25_Miner
from data_gen_prompts import *
from gen_utils import *
from lm_helper import OpenAILM, HFLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore

my_logger = MyLogger()


def hash_existed_documents(path):
    if not os.path.exists(path):
        return {}
    data = load_jsonl(path)
    hashed_data = {}
    for ex in data:
        query, document = ex['query'], ex['document']
        key = document
        if 'prompt' in ex:
            key += ex['prompt']
        hashed_data[key] = query
    return hashed_data


def check_sample_existence(document, hashed_data, prompt=None):
    key = document
    if prompt:
        key += prompt
    if key in hashed_data:
        return True, hashed_data[key]
    else:
        return False, None


def format_example(query, document, queries_per_doc, bm25_miner):
    decoder = json.JSONDecoder()
    try:
        json_data, _ = decoder.raw_decode(query[query.find('{'):])
        if 'hard_query' in json_data:
            queries = json_data['hard_query']
        else:
            queries = json_data['questions']
        if len(queries) > queries_per_doc:
            queries = queries[:queries_per_doc]
    except Exception as e:
        my_logger.error(e)
        my_logger.error(f"Skipping query: {query}")
        return None
    items = []

    my_logger.info("Start making items")
    for _query in queries:
        my_logger.info(f"Query: {query}")
        pos = document
        try:
            if isinstance(_query, dict):
                _question = _query['question']
                _scenario = _query['scenario']
                _question = f"{_scenario} {_question}"
            else:
                _question = _query
            negs = bm25_miner.select_hard_negatives(_question, pos, 1)
        except Exception as e:
            my_logger.error(e)
            my_logger.error(f"Skipping query with type {type(_query)}: {_query}")
            continue
        item = {
            'query': _question,
            'pos': [pos],
            'neg': negs,
        }
        my_logger.info(f"Generated item: {item}", 1)
        items.append(item)
    return items


def shuffle_paired_lists(a, b):
    pairs = list(zip(a, b))
    random.shuffle(pairs)

    a_shuffled, b_shuffled = zip(*pairs)

    return list(a_shuffled), list(b_shuffled)


def doc2query(bm25_miner, subject: str, pids, qid_to_pids,
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", 
                num_docs=100, queries_per_doc=1, filter_name=None, 
                output_dir='synthetic_data', prompt_id='hq_gen', 
                num_prompts=1, temperature=0, top_p=0):

    prompt = prompt_registry[prompt_id]

    documents, doc_ids = bm25_miner.documents, bm25_miner.doc_ids
    doc_dicts = [{'doc_id': doc_id, 'doc': doc} for doc_id, doc in zip(doc_ids, documents)]

    total_num_docs = len(doc_dicts)
    # num_docs_sample_pool = min(num_docs*1, total_num_docs)  # document pool to sample num_oversample_docs docs
    # num_oversample_docs = min(num_docs*2, total_num_docs)  # obtain more documents than expected to make the final output matches the expectation
    filter_cache_dir = f'cache/{subject}'
    os.makedirs(filter_cache_dir, exist_ok=True)
    my_logger.info(f"Filtering documents based on {filter_name}...")

    doc_dicts, doc_ids = document_filter(doc_dicts, doc_ids, pids, filter_name=filter_name, num_docs=num_docs, cache_dir=filter_cache_dir)
    # doc_dicts, doc_ids = shuffle_paired_lists(doc_dicts, doc_ids)

    num_filtered_docs = len(doc_dicts)

    my_logger.info(f"Total number of documents after filtering: {total_num_docs}")
    my_logger.info(f"Number of filtered documents with oversampling: {num_filtered_docs}")

    model_id_str = model_id.split('/')[-1]
    # path to save intermediate model generated results, will check if the same document has been used before to avoid repetitive generation.
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_output_path = os.path.join(output_dir, f'all_docs_train_data/{prompt_id}/{model_id_str}/{subject}_{num_docs}_train_data_{timestamp}.jsonl')
    final_output_path = os.path.expanduser(final_output_path)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    output_path = os.path.join(output_dir, f'all_docs/{prompt_id}/{model_id_str}/{subject}_{num_docs}_train_data_{timestamp}.jsonl')
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hashed_data = hash_existed_documents(output_path)

    if 'gpt' in model_id:
        model = OpenAILM(model_id, temperature=temperature, top_p=top_p, seed=0)
    else:
        model = HFLM(model_id, temperature=temperature, top_p=top_p)

    system_prompt = fill_sys_prompt(prompt, queries_per_doc=queries_per_doc)
    system_prompts = [system_prompt]
    
    final_training_data = []
    my_logger.info(f"Sampling from system prompts:")
    for system_prompt in system_prompts:
        my_logger.info(system_prompt, 1)
    my_logger.debug("End of system prompts.")
    my_logger.info()

    # Avoid generate using pid if any of its sibling has generated
    # Analogy
    # qid: parent
    # pids: children
    forbidden_pids = []
    generated_qids = get_generated_qids(subject)

    with open(output_path, 'a+', buffering=1) as fout:
        for doc_dict in tqdm(doc_dicts):
            document = doc_dict['doc']
            doc_id = doc_dict['doc_id']

            if doc_id in forbidden_pids:
                continue

            pairs_used_pid = get_pairs_used_pid(qid_to_pids, doc_id)
            for pair in pairs_used_pid:
                pair_qid = pair["q_id"]
                pids = qid_to_pids[pair_qid]
                forbidden_pids.extend(pids)
            
            pairs_used_pid = [pair for pair in pairs_used_pid if pair["q_id"] not in generated_qids]

            add_to_record(subject, pairs_used_pid)

            formatted_query = format_query_doc(document)

            if len(system_prompts) > num_prompts:
                sampled_prompts = random.choices(system_prompts, k=num_prompts)
            else:
                sampled_prompts = system_prompts

            for system_prompt in sampled_prompts: # generate examples for each prompt
                my_logger.info()
                has_generated, query = check_sample_existence(document, hashed_data, prompt=system_prompt)
                if not has_generated or temperature > 0:
                    max_num_attempt_per_doc = 3
                    num_attempt = 0
                    succeed = False
                    while not succeed and num_attempt < max_num_attempt_per_doc:
                        query = model.generate(formatted_query, system_prompt=system_prompt)

                        my_logger.info(f"Generated query: {query}", 1)
                        my_logger.info()
                        items = {'query': query, 'document': document, 'prompt': system_prompt}
                        items = format_example(query, document, queries_per_doc, bm25_miner)
                        num_attempt += 1
                        if items is not None:
                            succeed = True
                        if not succeed:
                            my_logger.debug(f"Not success, remaining attempts: {max_num_attempt_per_doc-num_attempt}")
                        else:
                            my_logger.debug(f"Success, prepare to write result.")
                else:
                    my_logger.info(f"Using existing query. Skipping.", 1)
                    my_logger.info()
                if not items:
                    my_logger.info(f"Skipping document: {document}", 1)
                    my_logger.info()
                    continue
                if isinstance(items, list):
                    if len(items) < queries_per_doc:
                        continue
                    final_training_data.extend(items[:queries_per_doc])
            if len(final_training_data) >= num_docs * queries_per_doc:
                break
            my_logger.info()
    
    write_jsonl(final_training_data, final_output_path)


def get_pairs_used_pid(qid_to_pids: dict[str, List[str]], target_pid: str) -> List[dict[str, str]]:
    # Get {qid, pid} pairs that used the given pid
    result = []
    for qid, pids in qid_to_pids.items():
        if target_pid in pids:
            result.append({"q_id": qid, "p_id": target_pid})
    return result


def get_generated_qids(dataset: str) -> set[str]:
    # Read from past generated files
    # Make a list of qids that has been generated
    result = set()
    os.makedirs("outputs/generation_record", exist_ok=True)
    folder = Path("outputs/generation_record")
    files = [p for p in folder.iterdir() if p.name.startswith(dataset)]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    content = json.loads(line)
                    result.add(content["q_id"])
    return result


def add_to_record(dataset: str, qid_to_chosen_pid: List[dict[str, str]]):
    save_path = os.path.join("outputs/generation_record", dataset)
    save_path += ".jsonl"
    with open(save_path, "a", encoding="utf-8") as f:
        for item in qid_to_chosen_pid:
            f.write(json.dumps(item) + "\n")


def get_pids(qid_to_pids: dict[str, List[str]], dataset: str) -> List[str]:
    result = []
    generated_qids = get_generated_qids(dataset)
    for qid, pids in qid_to_pids.items():
        if qid not in generated_qids:  # Remove this line if you don't want to enforce uniqueness rule
            result.extend(pids)

    return result


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


def read_fill_file() -> List[Tuple[str, int]]:
    result = []
    fill_file_path = "outputs/fill.txt"
    with open(fill_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace("q_id: ", "")
            line = line.split(", len: ")
            result.append((line[0], int(line[1])))
    return result


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default=None, help='mode')
    parser.add_argument('--dataset', type=str, default='MedicalSciences', help='the dataset, see README')
    parser.add_argument('--model_id', type=str, default='gpt-4o', help='model id')
    # parser.add_argument('--dataset', type=str, default='bright', help='dataset')
    # parser.add_argument('--subject', type=str, default=None, help='subject')
    parser.add_argument('--queries_per_doc', type=int, default=3, help='number of generated samples per document')
    parser.add_argument('--num_docs', type=int, default=None, help='number of documents to sample for each subject')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--filter', type=str, default=None, help='the default filter is the length filter', choices=['length', 'fineweb', 'dclm'])
    # parser.add_argument('--data_path', type=str, default='~/data/chunks/mathpile_wiki_chunks.jsonl', help='data path')
    parser.add_argument('--output_dir', type=str, default='outputs/synthetic_questions', help='base directory to save the generated data')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='cache directory to save cached data during document filtering.')
    parser.add_argument('--prompt_id', type=str, default='hq_gen', help='prompt to use')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    # args.data_path = os.path.expanduser(args.data_path)
    os.makedirs(args.cache_dir, exist_ok=True)
    # print(args)


    documents: List[str] = None
    doc_ids: List[str] = None

    qid_to_pids: dict[str, List[str]] = None

    match args.dataset:
        case "MedicalSciences":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = MedicalSciencesDataReader().get_documents()
            qid_to_pids = MedicalSciencesQrelDataReader().get_qid_to_pids()
        case "PMCTreatment":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = PMCTreatmentDataReader().get_documents()
            qid_to_pids = PMCTreatmentQrelDataReader().get_qid_to_pids()
        case "IIYiClinical":
            my_logger.info(f"Loading dataset: {args.dataset}")
            documents, doc_ids = IIYiClinicalDataReader().get_documents()
            qid_to_pids = IIYiClinicalQrelDataReader().get_qid_to_pids()
        case _:
            my_logger.error("Invalid dataset. Please see README for valid datasets.")

    # bm25_miner = BM25_Miner(documents, doc_ids)
    # passages = []
    # for document, doc_id in zip(documents, doc_ids):
    #     if doc_id in pids:
    #         passages.append(doc_id)

    bm25_miner = BM25_Miner(documents, doc_ids)

    pids = get_pids(qid_to_pids, args.dataset)

    num_docs = args.num_docs
    fill = read_fill_file()
    valid_num_docs = 0
    if len(fill) != 0:
        generation_record = get_generation_record(args.dataset)
        new_qid_to_pids = dict()
        new_pids = []
        for item in fill:
            qid = item[0]

            should_skip = False
            for pair in generation_record:
                if pair["q_id"] == qid:
                    should_skip = True
            if should_skip: continue

            new_qid_to_pids[qid] = qid_to_pids[qid]
            valid_num_docs += 1

            pids_for_qid = qid_to_pids[qid]
            new_pids.extend(pids_for_qid)
        num_docs = valid_num_docs
        for pid in new_pids:
            if pid not in pids:
                new_pids.remove(pid)
        pids = new_pids
        qid_to_pids = new_qid_to_pids

    model_id = args.model_id
    doc2query(bm25_miner, subject=args.dataset, pids=pids, qid_to_pids=qid_to_pids,
                model_id=model_id, num_docs=num_docs, filter_name=args.filter, 
                queries_per_doc=args.queries_per_doc, output_dir=args.output_dir, 
                prompt_id=args.prompt_id, temperature=args.temperature, top_p=args.top_p)


if __name__ == '__main__':
    main()
