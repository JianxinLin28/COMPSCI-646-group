python run.py --mode eval_retrieval --task Medical-Sciences --retriever_name bm25 --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task Medical-Sciences --retriever_name medcpt --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task Medical-Sciences --retriever_name bge --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task Medical-Sciences --retriever_name contriever --mode eval_retrieval --reranker_name bge-reranker

python run.py --mode eval_retrieval --task PMC-Treatment --retriever_name bm25 --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task PMC-Treatment --retriever_name medcpt --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task PMC-Treatment --retriever_name bge --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task PMC-Treatment --retriever_name contriever --mode eval_retrieval --reranker_name bge-reranker

python run.py --mode eval_retrieval --task IIYi-Clinical --retriever_name bm25 --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task IIYi-Clinical --retriever_name medcpt --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task IIYi-Clinical --retriever_name bge --mode eval_retrieval --reranker_name bge-reranker
python run.py --mode eval_retrieval --task IIYi-Clinical --retriever_name contriever --mode eval_retrieval --reranker_name bge-reranker