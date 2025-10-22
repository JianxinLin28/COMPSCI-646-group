from typing import List
from datasets import load_dataset # type: ignore
from pyserini import analysis # type: ignore
from gensim.corpora import Dictionary # type: ignore
from gensim.models import LuceneBM25Model # type: ignore
from gensim.similarities import SparseMatrixSimilarity # type: ignore
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore

my_logger= my_logger.MyLogger()

class BM25_Miner():
    def __init__(self, 
                documents: List[str], 
                doc_ids: List[str]):
        self.documents = documents
        self.doc_ids = doc_ids
        self.hashed_documents = self.get_hashed_documents(documents, doc_ids)
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        corpus = [self.analyzer.analyze(x) for x in documents]
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
                                                normalize_queries=False, normalize_documents=False)

    def get_hashed_documents(self, documents, doc_ids):
        hashed_documents = {}
        for docid, doc in zip(doc_ids, documents):
            hashed_documents[docid] = doc
        return hashed_documents

    def select_hard_negatives(self, query, gold_doc, num_neg=1, hard_neg_start_index=20):
        scores = self.search(query)
        
        num_added = 0
        hard_negatives_ids = []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (doc_id, score) in enumerate(sorted_scores):
            if i >= hard_neg_start_index:
                # avoid selecting false negative
                if self.hashed_documents[doc_id] != gold_doc:
                    hard_negatives_ids.append(doc_id)
                    num_added += 1
            if num_added == num_neg:
                break

        hard_negative_documents = self.get_documents_text(hard_negatives_ids)
        return hard_negative_documents

    def search(self, query):
        query = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(query)]
        similarities = self.bm25_index[bm25_query].tolist()
        all_scores = {}
        for did, s in zip(self.doc_ids, similarities):
            all_scores[did] = s
        cur_scores = sorted(all_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores = {}
        for pair in cur_scores:
            all_scores[pair[0]] = pair[1]
        return all_scores

    def get_documents_text(self, docids):
        return [self.hashed_documents[docid] for docid in docids]
