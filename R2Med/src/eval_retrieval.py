import os
from time import time
from collections import defaultdict
import json
from mteb.abstasks.AbsTaskRetrieval import DRESModel
DRES_METHODS = ["encode_queries", "encode_corpus"]
from utils import FlagDRESModel, InstructorModel, BiEncoderModel, HighScaleModel, GritModel, NVEmbedModel, RetrievalOPENAI
from instrcution import *
from tqdm import tqdm
from base import split_response_for_r1, r1_model_list, writejson_bench
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import pickle as pkl

def is_hard_negative(doc_id):
    pid_pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    
    return pid_pattern.match(doc_id)


def visualize_query_neighborhood_pca(
    model,
    corpus,
    queries,
    relevant_docs,
    hard_negative,
    retrieved,
    query_id,
    out_path,
    max_other_hn=None,   # cap for other queries' HNs (None = use all)
    batch_size=128,
):
    """
    Visualize one query + neighbors via PCA (2D):

    - Query:                      red
    - Ground-truth docs:          green
    - Hard negatives (this q):    black
    - Hard negatives (other q's): purple
    - Other retrieved docs:       blue

    `model` is expected to be the DRES(...) wrapper from beir, so we
    grab its underlying encoder via `model.model`.
    """
    if query_id not in queries:
        print(f"[PCA] query_id {query_id} not found in queries. Skip.")
        return

    # Use the underlying encoder if this is a DRES wrapper
    encoder = getattr(model, "model", model)

    # ---- 1) Get query text -----------------------------------------------
    q_raw = queries[query_id]
    if isinstance(q_raw, str):
        query_text = q_raw
    elif isinstance(q_raw, (list, tuple)):
        # your code sometimes stores [e['text'], e['hy_doc']]
        query_text = q_raw[0]
    else:
        query_text = str(q_raw)

    # ---- 2) Positives and this-query HNs --------------------------------
    pos_ids = list(relevant_docs.get(query_id, {}).keys())
    hn_ids = list(hard_negative.get(query_id, []))
    
    pos_set = set(retrieved).intersection(set(pos_ids))
    hn_set = set(retrieved).intersection(set(hn_ids))
    
    pos_ids = list(pos_set)
    hn_ids = list(hn_set)

    # ---- 3) All other hard negatives (union over all queries) -----------
    all_hn_ids = set()
    for qid, pids in hard_negative.items():
        for pid in pids:
            if pid in retrieved:
                all_hn_ids.add(pid)

    other_hn_ids = list(all_hn_ids - hn_set)

    if max_other_hn is not None and len(other_hn_ids) > max_other_hn:
        other_hn_ids = random.sample(other_hn_ids, max_other_hn)

    other_hn_set = set(other_hn_ids)

    # ---- 4) Other retrieved docs (instead of random corpus docs) --------
    # `retrieved` is results[query_id], usually a dict {doc_id: score}
    if isinstance(retrieved, dict):
        retrieved_ids = list(retrieved.keys())
    else:
        retrieved_ids = list(retrieved)

    # Blue docs = retrieved docs that are not positives or any HN
    blue_candidates = [
        pid for pid in retrieved_ids
        if pid not in pos_set and pid not in hn_set and pid not in other_hn_set
    ]
    print('blue_candidates:', len(blue_candidates))
    blue_ids = blue_candidates

    # Nothing to plot?
    if (
        len(pos_ids) == 0
        and len(hn_ids) == 0
        and len(other_hn_ids) == 0
        and len(blue_ids) == 0
    ):
        print(f"[PCA] No docs to visualize for query_id {query_id}.")
        return

    # ---- 5) Filter out any IDs not in corpus ----------------------------
    pos_ids = [pid for pid in pos_ids if pid in corpus]
    hn_ids = [pid for pid in hn_ids if pid in corpus]
    other_hn_ids = [pid for pid in other_hn_ids if pid in corpus]
    blue_ids = [pid for pid in blue_ids if pid in corpus]

    # If all got filtered out, bail
    if (
        len(pos_ids) == 0
        and len(hn_ids) == 0
        and len(other_hn_ids) == 0
        and len(blue_ids) == 0
    ):
        print(f"[PCA] No valid corpus docs to visualize for query_id {query_id}.")
        return

    # ---- 6) Encode query and documents ----------------------------------
    # Query
    query_emb = np.array(
        encoder.encode_queries([query_text], batch_size=batch_size)
    )  # [1, d]

    # Docs in fixed order: [positives, this-q HNs, other HNs, retrieved others]
    doc_ids = pos_ids + hn_ids + other_hn_ids + blue_ids
    doc_dicts = [corpus[pid] for pid in doc_ids]

    doc_embs = np.array(
        encoder.encode_corpus(doc_dicts, batch_size=batch_size)
    )  # [N_docs, d]

    # ---- 7) PCA to 2D ----------------------------------------------------
    all_embs = np.concatenate([query_emb, doc_embs], axis=0)  # [1+N, d]
    
    # pca = PCA(n_components=2)
    # all_2d = pca.fit_transform(all_embs)
    X_scaled = StandardScaler().fit_transform(all_embs)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
    )
    all_2d = tsne.fit_transform(X_scaled)  # shape (150, 2)

    q_2d = all_2d[0]
    docs_2d = all_2d[1:]

    n_pos = len(pos_ids)
    n_hn = len(hn_ids)
    n_other_hn = len(other_hn_ids)
    n_blue = len(blue_ids)

    pos_2d = docs_2d[:n_pos]
    hn_2d = docs_2d[n_pos:n_pos + n_hn]
    other_hn_2d = docs_2d[n_pos + n_hn:n_pos + n_hn + n_other_hn]
    blue_2d = docs_2d[n_pos + n_hn + n_other_hn:n_pos + n_hn + n_other_hn + n_blue]

    # ---- 8) Plot with requested colors ----------------------------------
    plt.figure(figsize=(6, 6))

    # Query: red
    plt.scatter(q_2d[0], q_2d[1], c='r', s=80, marker='x', label='query')

    # Positives: green
    if len(pos_2d) > 0:
        plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='g', s=25, alpha=0.8, label='relevant docs')

    # This-query HNs: black
    if len(hn_2d) > 0:
        plt.scatter(hn_2d[:, 0], hn_2d[:, 1], c='k', s=25, alpha=0.8, label='hard negatives (this q)')

    # Other queries' HNs: purple
    if len(other_hn_2d) > 0:
        plt.scatter(other_hn_2d[:, 0], other_hn_2d[:, 1], c='m', s=20, alpha=0.6,
                    label='hard negatives (other q)')

    # Other retrieved docs: blue
    if len(blue_2d) > 0:
        plt.scatter(blue_2d[:, 0], blue_2d[:, 1], c='b', s=15, alpha=0.5, label='other retrieved')

    plt.title(f"Query neighborhood (PCA) – qid={query_id}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[PCA] Saved query neighborhood PCA to {out_path}")



def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else -1


def is_dres_compatible(model):
    for method in DRES_METHODS:
        op = getattr(model, method, None)
        if not (callable(op)):
            return False
    return True

def load_retrieval_data(hf_hub_name="", gar_method="", gar_llm="", r1_mode=False):
    corpus = defaultdict(dict)
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['id']
            corpus[pid] = {"text": e['text']}
    queries = {}
    if gar_method != "" and gar_llm != "":
        query_path = hf_hub_name + f"/{gar_method}/{gar_llm}/query_with_hydoc.jsonl"
    else:
        query_path = hf_hub_name + f"/query.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            qid = e['id']
            if gar_method != "" and gar_llm != "":
                if r1_mode:
                    queries[qid] = [e['text'], split_response_for_r1(gar_llm, e["hy_doc"])]
                else:
                    if gar_method == "query2doc":
                        queries[qid] = e['text'] + "[SEP]" + e["hy_doc"]
                    else:
                        queries[qid] = [e['text'], e["hy_doc"]]
            else:
                queries[qid] = e['text']

    relevant_docs = defaultdict(dict)
    with open(hf_hub_name + "/qrels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['p_id']
            qid = e['q_id']
            relevant_docs[qid][pid] = int(e["score"])
    qrels_num = 0
    for k, v in relevant_docs.items():
        qrels_num += len(v)
    print(f"共{len(queries)}个queries,{len(corpus)}个文章！,{qrels_num}个相关数据！")
    
    
    """
    Load Hard Negative Documents
    """
    dataset_name = hf_hub_name.split('/')[-2]
    dataset_name = ''.join(dataset_name.split('-'))
    
    hn_corpus = {}
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
                hn_corpus[e['hard_passage_id']] = {"text": e['hard_passage']}
    
    items = list(hn_corpus.items())
    random.shuffle(items)

    hn_corpus = dict(items)
    print('number of hard negative documents:', len(hn_corpus))
    
    corpus.update(hn_corpus)
    
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

    
    return corpus, queries, relevant_docs, hard_negative

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

def retrieval_by_dense(
        queries=None,
        corpus=None,
        relevant_docs=None,
        hard_negative=None,
        model=None,
        batch_size=512,
        corpus_chunk_size=None,
        score_function="cos_sim",
        **kwargs
):
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")
    corpus, queries = corpus, queries
    model = model if is_dres_compatible(model) else DRESModel(model)
    
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    model = DRES(
        model,
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 100000,
        **kwargs,
    )
    retriever = EvaluateRetrieval(model, k_values=[1, 10, 25, 50, 100],
                                  score_function=score_function)  # or "cos_sim" or "dot"

    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values,
                                                       ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    print(scores)
    for query_id in queries:
        visualize_query_neighborhood_pca(
            model=model,
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            hard_negative=hard_negative,
            retrieved=results[query_id],
            query_id=query_id,
            out_path=f'./img/BGE_MS_{query_id}.png',
            batch_size=128,
        )
            
    
    
    for k in retriever.k_values:
        avg_hn_rate = []
        avg_true_hn_rate = []
        avg_has_hn_rate = []
        avg_has_true_hn_rate = []
        avg_hn_rr = []
        avg_true_hn_rr = []
        
        for query_id in queries:
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
            with open('./hnrr_bge_medical_k100.pkl', 'wb') as f:
                pkl.dump(avg_hn_rr, f)
            with open('./true_hnrr_bge_medical_k100.pkl', 'wb') as f:
                pkl.dump(avg_true_hn_rr, f)
            # print('number of hn_RR:', len(avg_hn_rr))
            # print('number of True_hn_RR:', len(avg_true_hn_rr))
            # plt.hist(avg_hn_rr, bins=20)
            # plt.savefig('HNRR_MedCPT_MedicalSciences_k100.png')
            # plt.close()
            
            # plt.hist(avg_true_hn_rr, bins=20)
            # plt.savefig('True_HNRR_MedCPT_MedicalSciences_k100.png')
            # plt.close()
            
        # compute means
        print(f'has_hn@{k}: {np.sum(avg_has_hn_rate)}/{len(avg_has_hn_rate)}')
        print(f'has_hn@{k}: {np.sum(avg_has_true_hn_rate)}/{len(avg_has_true_hn_rate)}')
        avg_hn_rate = safe_mean(avg_hn_rate)
        avg_true_hn_rate = safe_mean(avg_true_hn_rate)
        avg_has_hn_rate = safe_mean(avg_has_hn_rate)
        avg_has_true_hn_rate = safe_mean(avg_has_true_hn_rate)
        avg_hn_rr = safe_mean(avg_hn_rr)
        avg_true_hn_rr = safe_mean(avg_true_hn_rr)

        # print (optional)
        print(f'has_hn_rate@{k}: {avg_has_hn_rate:.4f}')
        print(f'has_true_hn_rate@{k}: {avg_has_true_hn_rate:.4f}')
        print(f'hn_rate@{k}: {avg_hn_rate:.4f}/{k} = {avg_hn_rate/k:.4f}')
        print(f'true_hn_rate@{k}: {avg_true_hn_rate:.4f}/{k} = {avg_true_hn_rate/k:.4f}')
        print(f'hn_rr@{k}: {avg_hn_rr:.4f}')
        print(f'true_hn_rr@{k}: {avg_true_hn_rr:.4f}')
        print()

        # --------------------------------------
        # ✅ ADD NEW METRICS INTO `scores`
        # --------------------------------------
        scores.update({
            f"has_hn_rate_at_{k}": avg_has_hn_rate,
            f"has_true_hn_rate_at_{k}": avg_has_true_hn_rate,
            f"hn_rate_at_{k}": avg_hn_rate,
            f"hn_rate_ratio_at_{k}": avg_hn_rate / k,
            f"true_hn_rate_at_{k}": avg_true_hn_rate,
            f"true_hn_rate_ratio_at_{k}": avg_true_hn_rate / k,
            f"hn_rr_at_{k}": avg_hn_rr,
            f"true_hn_rr_at_{k}": avg_true_hn_rr,
        })
    
    
    return scores, results


def init_model(retriever_name, retrieral_model_path):
    if retriever_name in ["contriever"]:
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                        pooling_method="mean",
                                        normalize_embeddings=False)
    elif retriever_name in ["inst-l", "inst-xl"]:
        retrieval_model = InstructorModel(model_name_or_path=retrieral_model_path,
                                      query_instruction_for_retrieval="",
                                      document_instruction_for_retrieval="",
                                      batch_size=64 if retriever_name == "inst-l" else 32)
    elif retriever_name in ["e5", "sfr"]:
        retrieval_model = HighScaleModel(model_name_or_path=retrieral_model_path,
                                        query_instruction_for_retrieval=query_instruction[retriever_name],
                                        pooling_method='last',
                                        max_length=2048,
                                        batch_size=4)
    elif retriever_name in ["bge"]:
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                  query_instruction_for_retrieval=query_instruction[retriever_name],
                                  pooling_method='cls',
                                  max_length=512)
    elif retriever_name in ["bmr-410m", "bmr-2b"]:
        MAX_LENGTH = {"bmr-410m":512, "bmr-2b":1024}
        retrieval_model = FlagDRESModel(model_name_or_path=retrieral_model_path,
                                        encode_mode="BMR",
                                        query_instruction_for_retrieval="",
                                        pooling_method="last-bmr",
                                        max_length=MAX_LENGTH[retriever_name],
                                        document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=32)
    elif retriever_name in ["bmr-7b"]:
        retrieval_model = HighScaleModel(model_name_or_path=retrieral_model_path,
                                        encode_mode="BMR",
                                        query_instruction_for_retrieval="",
                                        pooling_method="last-bmr",
                                        max_length=2048,
                                        document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=64)
    elif retriever_name in ["medcpt"]:
        retrieval_model = BiEncoderModel(query_encoder_name_or_path=retrieral_model_path[1],
                                         doc_encoder_name_or_path=retrieral_model_path[1],
                                        max_length=512,
                                        batch_size=512)
    elif retriever_name in ["grit"]:
        retrieval_model = GritModel(model_name_or_path=retrieral_model_path,
                                      query_instruction_for_retrieval="",
                                     document_instruction_for_retrieval=doc_instruction[retriever_name],
                                        batch_size=64)
    elif retriever_name in ["nv"]:
        retrieval_model = NVEmbedModel(model_name_or_path=retrieral_model_path,
                                    query_instruction_for_retrieval="",
                                    document_instruction_for_retrieval=doc_instruction[retriever_name],
                                    batch_size=8,
                                    max_length=512)
    elif retriever_name in ["openai", "voyage"]:
        retrieval_model = RetrievalOPENAI(model_name_or_path=retrieral_model_path, batch_size=64)
    else:
        print(f"Please print a valid model name!")
        return None
    return retrieval_model

def eval_retrieval(retriever_name="", gar_method="", gar_llm="", task_names=[""]):

    r1_mode = False
    if gar_llm in r1_model_list:
        r1_mode = True
    model_path_dict = {
        "contriever": "../model_dir/contriever-msmarco",
        "medcpt": ["../model_dir/MedCPT-Query-Encoder",
                   "../model_dir/MedCPT-Article-Encoder"],
        "inst-l": "model_dir/hkunlp/instructor-large",
        "inst-xl": "model_dir/hkunlp/instructor-xl",
        "bmr-410m": "model_dir/BMRetriever/BMRetriever-410M",
        "bmr-2b": "model_dir/BMRetriever/BMRetriever-2B",
        "bmr-7b": "model_dir/BMRetriever/BMRetriever-7B",
        "bge": "../model_dir/bge-large-en-v1.5",
        "e5": "../model_dir/e5-mistral-7b-instruct",
        "grit": "model_dir/GritLM/GritLM-7B",
        "sfr": "model_dir/Salesforce/SFR-Embedding-Mistral",
        "nv": "../model_dir/NV-Embed-v2",
        "openai": "text-embedding-3-large",
        "voyage": "voyage-3",
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    retrieral_model_path = model_path_dict[retriever_name]
    retrieval_model = init_model(retriever_name, retrieral_model_path)
    print(f"Embedding model have been loaded from {retrieral_model_path}")
    t00 = time()
    ndcg_values = []
    save_top_k = False # default to False
    for task_name in task_names:
        t0 = time()
        data_path = f'../dataset/{task_name}/'
        if gar_method != "":
            save_path = f"../results/gar/{gar_method}/{gar_llm}/{retriever_name}/{task_name}.json"
        else:
            save_path = f"../results/base retriever/{retriever_name}/{task_name}.json"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cache_path = f"../output/doc_embs/{retriever_name}/{task_name}.npy"
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        corpus, queries, relevant_docs, hard_negative = load_retrieval_data(data_path, gar_method, gar_llm, r1_mode)
        if retriever_name in ["e5", "bmr-410m", "bmr-2b", "bmr-7b","inst-l", "inst-xl", "grit", "sfr", "nv"]:
            retrieval_model.query_instruction_for_retrieval = query_instruction[retriever_name][task_name]
        if retriever_name in ["inst-l", "inst-xl"]:
            retrieval_model.document_instruction_for_retrieval = doc_instruction[retriever_name][task_name]
        if retriever_name in ["openai", "voyage"]:
            doc_cache_path = cache_path.replace(f"{task_name}", f"{task_name}-doc")
            retrieval_model.query_cache_path = cache_path
            retrieval_model.doc_cache_path = doc_cache_path
        else:
            retrieval_model.cache_path = cache_path
        scores, top_k_results = retrieval_by_dense(queries=queries, corpus=corpus, relevant_docs=relevant_docs, model=retrieval_model, hard_negative=hard_negative)
        evaluation_time = round((time() - t0) / 60, 2)
        task_results = {
            "dataset_name": task_name,
            "model_name": retriever_name,
            "evaluation_time": str(evaluation_time) + " minutes",
            "test": scores,
        }
        with open(save_path, "w") as f_out:
            json.dump(task_results, f_out, indent=2, sort_keys=True)
        ndcg_values.append(scores["ndcg_at_10"])
        print(f"{task_name} evaluation cost {evaluation_time} minutes!")
        if save_top_k:
            save_top_k_path = f"../output/topk_docs/{retriever_name}/{task_name}.jsonl"
            if not os.path.exists(os.path.dirname(save_top_k_path)):
                os.makedirs(os.path.dirname(save_top_k_path))
            save_topk_docs(top_k_results, save_top_k_path, 100)

    print(f"Model {retriever_name} evaluation all task cost {round((time() - t00) / 60, 2)} minutes!")
    ndcg_scaled = [round(value * 100, 2) for value in ndcg_values]
    print("\t".join(map(str, ndcg_scaled)))

