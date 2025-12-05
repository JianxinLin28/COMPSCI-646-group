import os
from time import time
from collections import defaultdict
from tqdm import tqdm
import json
import pytrec_eval
import logging
import random
logger = logging.getLogger(__name__)
from typing import List, Dict, Tuple
from base import split_response_for_r1, r1_model_list, writejson_bench
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def is_hard_negative(doc_id):
    pid_pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    
    return pid_pattern.match(doc_id)


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def calculate_retrieval_metrics(results, qrels, hard_negative, k_values=[1, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"mrr": 0}
    output = {**ndcg, **_map, **recall, **precision, **mrr}

    for k in k_values:
        ndcg[f"ndcg_at_{k}"] = 0.0
        _map[f"map_at_{k}"] = 0.0
        recall[f"recall_at_{k}"] = 0.0
        precision[f"precision_at_{k}"] = 0.0
        
        avg_hn_rate = []
        avg_true_hn_rate = []
        avg_has_hn_rate = []
        avg_has_true_hn_rate = []
        avg_hn_rr = []
        avg_true_hn_rr = []
        
        for query_id in qrels:
            hn_cnt = 0
            true_hn_cnt = 0
            hn_rr = -1
            true_hn_rr = -1

            for idx, doc_id in enumerate(results[query_id]):
                if idx >= k:
                    break
                if is_hard_negative(doc_id):
                    hn_cnt += 1
                    if hn_rr == -1:
                        hn_rr = idx
                        avg_hn_rr.append(hn_rr)
                    if doc_id in hard_negative[query_id]:
                        true_hn_cnt += 1
                        if true_hn_rr == -1:
                            true_hn_rr = idx
                            avg_true_hn_rr.append(true_hn_rr)

            # accumulate rates
            if hn_cnt > 0:
                avg_hn_rate.append(hn_cnt)
            if true_hn_cnt > 0:
                avg_true_hn_rate.append(true_hn_cnt)

            avg_has_hn_rate.append(1 if hn_cnt > 0 else 0)
            avg_has_true_hn_rate.append(1 if true_hn_cnt > 0 else 0)

        if k == 100:
            with open('./hnrr_bm25_medical_k100.pkl', 'wb') as f:
                pkl.dump(avg_hn_rr, f)
            with open('./true_hnrr_bm25_medical_k100.pkl', 'wb') as f:
                pkl.dump(avg_true_hn_rr, f)
        # compute means
        num_hn = np.sum(avg_has_hn_rate)
        num_true_hn = np.sum(avg_has_true_hn_rate)
        avg_hn_rate = safe_mean(avg_hn_rate)
        avg_true_hn_rate = safe_mean(avg_true_hn_rate)
        avg_has_hn_rate = safe_mean(avg_has_hn_rate)
        avg_has_true_hn_rate = safe_mean(avg_has_true_hn_rate)
        avg_hn_rr = safe_mean(avg_hn_rr)
        avg_true_hn_rr = safe_mean(avg_true_hn_rr)

        # print (optional)
        print(f'has_hn_rate@{k}: {num_hn}/{len(qrels)} = {avg_has_hn_rate:.4f}')
        print(f'has_true_hn_rate@{k}: {num_true_hn}/{len(qrels)} = {avg_has_true_hn_rate:.4f}')
        print(f'hn_rate@{k}: {avg_hn_rate:.4f}/{k} = {avg_hn_rate/k:.4f}')
        print(f'true_hn_rate@{k}: {avg_true_hn_rate:.4f}/{k} = {avg_true_hn_rate/k:.4f}')
        print(f'hn_rr@{k}: {avg_hn_rr:.4f}')
        print(f'true_hn_rr@{k}: {avg_true_hn_rr:.4f}')
        print()

        # --------------------------------------
        # ✅ ADD NEW METRICS INTO `scores`
        # --------------------------------------
        output.update({
            f"has_hn_rate_at_{k}": avg_has_hn_rate,
            f"has_true_hn_rate_at_{k}": avg_has_true_hn_rate,
            f"hn_rate_at_{k}": avg_hn_rate,
            f"hn_rate_ratio_at_{k}": avg_hn_rate / k,
            f"true_hn_rate_at_{k}": avg_true_hn_rate,
            f"true_hn_rate_ratio_at_{k}": avg_true_hn_rate / k,
            f"hn_rr_at_{k}": avg_hn_rr,
            f"true_hn_rr_at_{k}": avg_true_hn_rr,
        })

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)
    # print(f"scores length is {len(scores)}")
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"ndcg_at_{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"map_at_{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"recall_at_{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"precision_at_{k}"] += scores[query_id]["P_" + str(k)]
        mrr["mrr"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"ndcg_at_{k}"] = round(ndcg[f"ndcg_at_{k}"] / len(scores), 5)
        _map[f"map_at_{k}"] = round(_map[f"map_at_{k}"] / len(scores), 5)
        recall[f"recall_at_{k}"] = round(recall[f"recall_at_{k}"] / len(scores), 5)
        precision[f"precision_at_{k}"] = round(precision[f"precision_at_{k}"] / len(scores), 5)
    mrr["mrr"] = round(mrr["mrr"] / len(scores), 5)

    output.update({**ndcg, **_map, **recall, **precision, **mrr})
    return output

def retrieval_bm25(queries,query_ids,documents,doc_ids,qrels, hard_negative):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    results = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        results[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            results[str(query_id)][did] = s
        cur_scores = sorted(results[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:100]
        results[str(query_id)] = {}
        for pair in cur_scores:
            results[str(query_id)][pair[0]] = pair[1]
    scores = calculate_retrieval_metrics(results=results, qrels=qrels, hard_negative=hard_negative)
    return scores, results

def save_topk_docs(results, save_path, k):
    candidates = dict()
    for q_id in tqdm(results.keys()):
        can = results[q_id]
        sorted_can = dict(sorted(can.items(), key=lambda item: item[1], reverse=True))
        p_ids = [[d,s] for d, s in sorted_can.items()][:k]
        candidates[q_id] = p_ids
    new_data = []
    for id in candidates.keys():
        data = candidates[id]
        new_data.append(
            {
                "id": id,
                "topk_pid": data,
            }
        )
    writejson_bench(new_data, save_path)

def load_retrieval_data(hf_hub_name="", method = "", model_name="", r1_mode=False):
    doc_ids = []
    documents = []
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            doc_ids.append(e['id'])
            documents.append(e['text'])
    """
    Load Hard Negative Documents
    """
    dataset_name = hf_hub_name.split('/')[-2]
    dataset_name = ''.join(dataset_name.split('-'))
    hn_doc_ids = []
    hn_documents = []
    # hn_doc_path = f'{hf_hub_name}/hard_negative/qrel'
    hn_doc_path = '/home/xing/project/R2MED/dataset/hp'
    for fname in os.listdir(hn_doc_path):
        if not fname.startswith(dataset_name):
            continue
        if not fname.endswith("_hp_id_to_hp.jsonl"):
            continue
        hp_file = os.path.join(hn_doc_path, fname)
        with open(hp_file, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                hn_doc_ids.append(e['hard_passage_id'])
                hn_documents.append(e['hard_passage'])
    
    pairs = list(zip(hn_doc_ids, hn_documents))
    # shuffle in-place
    random.shuffle(pairs)

    # unzip back to two lists
    hn_doc_ids, hn_documents = zip(*pairs)

    print('number of normal documents:', len(doc_ids))
    print('number of hd documents:', len(hn_doc_ids))
    
    doc_ids += hn_doc_ids
    documents += hn_documents

    queries = []
    query_ids = []
    if method != "" and model_name != "":
        query_path = hf_hub_name + f"/{method}/{model_name}/query_with_hydoc.jsonl"
    else:
        query_path = hf_hub_name + f"/query.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            query_ids.append(e['id'])
            if method != "" and model_name != "":
                if r1_mode:
                    queries.append(split_response_for_r1(model_name, e["hy_doc"]))
                else:
                    if method == "query2doc":
                        queries.append(e["text"]*2+" "+e["hy_doc"])
                    elif method == "lamer":
                        queries.append(e["text"]+ " " + e["hy_doc"])
                    else:
                        queries.append(e["hy_doc"])
            else:
                queries.append(e["text"])

    ground_truth = defaultdict(dict)
    with open(hf_hub_name + "/qrels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['p_id']
            qid = e['q_id']
            ground_truth[qid][pid] = int(e["score"])
            
    hard_negative = {}
    for fname in os.listdir(hn_doc_path):
        if not fname.startswith(dataset_name):
            continue
        if not fname.endswith("_qrel_extent.jsonl"):
            continue
        hp_file = os.path.join(hn_doc_path, fname)
        with open(hp_file, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e['q_id'] not in hard_negative:
                    hard_negative[e['q_id']] = [e['p_id']]
                else:
                    hard_negative[e['q_id']].append(e['p_id'])
    
    # for key in hard_negative:
    #     print(f'query_id: {key}; number of ground truth: {len(ground_truth[key])}; number of hard negative: {len(hard_negative[key])}')

    return documents, doc_ids, queries, query_ids, ground_truth, hard_negative


def eval_bm25(gar_method="", gar_llm="", task_names=[""]):
    save_top_k = True # default False
    r1_mode = False
    if gar_llm in r1_model_list:
        r1_mode = True
    ndcg_values = []
    for task_name in task_names[:]:
        t0 = time()
        if gar_method != "":
            save_path = f"../results/gar/{gar_method}/{gar_llm}/bm25/{task_name}.json"
        else:
            save_path = f"../results/base retriever/bm25/{task_name}.json"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # if os.path.exists(save_path):
        #     print(f">>>WARNING: Model bm25 in dataset {task_name} results already exists. Skipping.")
        #     print()
        #     continue
        data_dir = f'../dataset/{task_name}/'
        documents, doc_ids, queries, query_ids, ground_truth, hard_negative = load_retrieval_data(data_dir, gar_method, gar_llm, r1_mode)
        scores, top_k_results = retrieval_bm25(queries, query_ids, documents, doc_ids, ground_truth, hard_negative)
        evaluation_time = round((time() - t0) / 60, 2)
        task_results = {
            "dataset_name": task_name,
            "model_name": "bm25",
            "evaluation_time": str(evaluation_time) + " minutes",
            "test": scores,
        }
        with open(save_path, "w") as f_out:
            json.dump(task_results, f_out, indent=2, sort_keys=True)
        ndcg_values.append(scores["ndcg_at_10"])
        if save_top_k:
            save_top_k_path = f"../output/topk_docs/bm25/{task_name}.jsonl"
            if not os.path.exists(os.path.dirname(save_top_k_path)):
                os.makedirs(os.path.dirname(save_top_k_path))
            save_topk_docs(top_k_results, save_top_k_path, 100)
    ndcg_scaled = [round(value * 100, 2) for value in ndcg_values]
    # print("\t".join(map(str, ndcg_scaled)))
    
    
if __name__ == '__main__':
    task_name = 'Medical-Sciences'
    data_dir = f'../dataset/{task_name}/'
    # dataset_name = data_dir.split('/')[-2]
    # dataset_name = ''.join(dataset_name.split('-'))
    # print(dataset_name)
    documents, doc_ids, queries, query_ids, ground_truth, hard_negative = load_retrieval_data(data_dir)
    # print(json.dumps(ground_truth, indent=2))