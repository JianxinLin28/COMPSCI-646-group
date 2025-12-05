<h1 align="center">R2MED: A Benchmark for Reasoning-Driven Medical Retrieval</h1>

This is the modified version of [https://github.com/R2MED/R2MED](https://github.com/R2MED/R2MED) for hard negative evaluation.

## ⚙️  Installation
Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

1. **Clone this repository:**

    ```bash
    git clone https://github.com/R2MDE/R2MED.git
    cd R2MED
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda create -n r2med python=3.10
    conda activate r2med
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install mteb==1.1.1
    pip install transformers==4.44.2
    pip install vllm==0.5.4
    ```

## 💾 Datasets Preparation

### Download the R2MED dataset:

- **R2MED:** [![HF Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow?style=flat-square)](https://huggingface.co/R2MED)

    Place all zip files under `./R2MED/dataset` and extract them.

### Data Structure:

For each dataset, the data is expected in the following structure:

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/project/R2MED/dataset/Biology
├── query.jsonl        # Query file
├── corpus.jsonl        # Document file
└── qrels.txt         # Relevant label file
```

## 💽 Evaluate
We evaluate 15 representative retrieval models of diverse sizes and architectures. Run the following command to get results:
```
cd ./src
python run.py --mode eval_retrieval --task {task} --retriever_name {retriever_name} --mode eval_retrieval --reranker_name bge-reranker
* `--task`: the task/dataset to evaluate. It can take one of `Medical-Sciences`,`PMC-Treatment`,`IIYi-Clinical`.
* `--retriever_name`: the retrieval model to evaluate. Current implementation supports `bm25`,`contriever`,`medcpt`,`bge`. \
```

## 📜Reference
```
@article{li2025r2med,
  title={R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  author={Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal={arXiv preprint arXiv:2505.14558},
  year={2025}
}
```
