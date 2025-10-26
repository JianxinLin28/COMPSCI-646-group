from datasets import DatasetDict # type: ignore
from typing import List, Tuple
import pandas as pd
from huggingface_hub import login # type: ignore
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore
from hf_dataset_manager import HFDatasetManager

# login()

my_logger= my_logger.MyLogger()

# ================================================
save_path = "./data"
paragraph_sep = "@@\n"
# ================================================


class DataReader:
    def __init__(self, hf_manager: HFDatasetManager):
        self.hf_manager = hf_manager  # must be loaded
    
    def get_column_names(self):
        return self.hf_manager.dataset.column_names

    def save_arrow(self, output_dir: str):
        self.hf_manager.save(output_dir)

    def column_to_list(self, split: str, column: str) -> List[str]:
        dataset = self.hf_manager.dataset

        if isinstance(dataset, DatasetDict):
            if split in dataset:
                ds = self.hf_manager.dataset[split]
                return [str(x) for x in ds[column]]
            else:
                my_logger.error(f"No split: {split}.")
            return []
        else:
            my_logger.error("Missing dataset.")
            return []


def read_first_100_medical_sciences():
    df = pd.read_json("hf://datasets/R2MED/Medical-Sciences/corpus.jsonl", lines=True)

    subset = df["text"].head(100)

    os.makedirs(save_path or ".", exist_ok=True)
    text_file = os.path.join(save_path, "Medical-Sciences_text_100.txt")
    id_file = os.path.join(save_path, "Medical-Sciences_id_100.txt")

    with open(text_file, "w", encoding="utf-8") as f:
        for line in subset:
            # Handle possible NaN values
                if isinstance(line, str):
                    f.write(line.strip() + "\n")
                    f.write(paragraph_sep)
                else:
                    f.write("\n")  # blank line if value not string
                    f.write(paragraph_sep)

    subset = df["id"].head(100)
    with open(id_file, "w", encoding="utf-8") as f:
        for line in subset:
            # Handle possible NaN values
                if isinstance(line, str):
                    f.write(line.strip() + "\n")
                    f.write(paragraph_sep)
                else:
                    f.write("\n")  # blank line if value not string
                    f.write(paragraph_sep)

def save_to(file_name: str, paragraphs: List[str]):
    os.makedirs(save_path or ".", exist_ok=True)
    text_file = os.path.join(save_path, file_name)
    
    with open(text_file, "w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(p.strip() + "\n")
            f.write(paragraph_sep)

def get_first_100_medical_sciences() -> Tuple[List[str], List[str]]:
    text_file = os.path.join(save_path, "Medical-Sciences_text_100.txt")
    id_file = os.path.join(save_path, "Medical-Sciences_id_100.txt")
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()
        paragraphs = content.split(paragraph_sep)
        paragraphs = [paragraph.strip() for paragraph in paragraphs]

    with open(id_file, "r", encoding="utf-8") as f:
        content = f.read()
        p_ids = content.split(paragraph_sep)
        p_ids = [p_id.strip() for p_id in p_ids]
    return paragraphs, p_ids


def main():
    # get_first_100_medical_sciences()
    # documents, doc_ids = get_first_100_medical_sciences()
    # for document in documents:
    #     my_logger.info(document)
    #     my_logger.info()

    def test_load_from_hf():
        hf_manager = HFDatasetManager("hf://datasets/R2MED/Medical-Sciences/corpus.jsonl")
        hf_manager.load()
        data_reader = DataReader(hf_manager)
        data_reader.save_arrow(os.path.join(save_path, "MedicalSciences"))

    def test_load_local_arrow():
        hf_manager = HFDatasetManager(os.path.join(save_path, "MedicalSciences"))
        hf_manager.load()
        data_reader = DataReader(hf_manager)
        data_reader.hf_manager.info()

    def sample_texts():
        hf_manager = HFDatasetManager(os.path.join(save_path, "MedicalSciences"))
        hf_manager.load()
        data_reader = DataReader(hf_manager)
        samples = data_reader.column_to_list("train", "text")[:2]
        for sample in samples:
            my_logger.info(sample)
            my_logger.info()

    # test_load_from_hf()
    # test_load_local_arrow()
    sample_texts()


if __name__ == "__main__":
    main()
