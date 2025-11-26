import requests
from datasets import DatasetDict # type: ignore
from typing import List, Tuple
from huggingface_hub import login # type: ignore
import sys
import os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore
from hf_dataset_manager import HFDatasetManager

# login()

my_logger= my_logger.MyLogger()

# ================================================
save_path = "./data"
save_path_qrel = "./data_qrel"
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
        
    def read_rows(self, split: str) -> List[dict]:
        dataset = self.hf_manager.dataset

        if isinstance(dataset, DatasetDict):
            if split in dataset:
                ds = self.hf_manager.dataset[split]
                return ds.to_list()
            else:
                my_logger.error(f"No split: {split}.")
            return []
        else:
            my_logger.error("Missing dataset.")
            return []


class BaseHFDataReader:
    def __init__(self):
        self.hf_path: str
        self.name:str
        self.save_dir: str
        self.hf_manager: HFDatasetManager = None
        self.data_reader: DataReader = None

    def _load(self):
        if os.path.exists(self.save_dir):
            self._load_downloaded()
        else:
            self._load_new()

    def _load_new(self):
        self.hf_manager = HFDatasetManager(self.hf_path)
        self.hf_manager.load()
        self.data_reader = DataReader(self.hf_manager)
        self.data_reader.save_arrow(self.save_dir)

    def _load_downloaded(self):
        self.hf_manager = HFDatasetManager(self.save_dir)
        self.hf_manager.load()
        self.data_reader = DataReader(self.hf_manager)

    def get_documents(self) -> Tuple[List[str], List[str]]:
        paragraphs = self.data_reader.column_to_list("train", "text")
        pids = self.data_reader.column_to_list("train", "id")
        return paragraphs, pids


class MedicalSciencesDataReader(BaseHFDataReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/Medical-Sciences/corpus.jsonl"
        self.name = "MedicalSciences"
        self.save_dir = os.path.join(save_path, self.name)
        self._load()


class PMCTreatmentDataReader(BaseHFDataReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/PMC-Treatment/corpus.jsonl"
        self.name = "PMCTreatment"
        self.save_dir = os.path.join(save_path, self.name)
        self._load()


class IIYiClinicalDataReader(BaseHFDataReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/IIYi-Clinical/corpus.jsonl"
        self.name = "IIYiClinical"
        self.save_dir = os.path.join(save_path, self.name)
        self._load()


class BaseHFDataQrelReader(BaseHFDataReader):
    def get_qid_to_pids(self) -> dict[str, List[str]]:
        result = dict()
        rows = self.data_reader.read_rows("train")

        for row in rows:
            qid = row["q_id"]
            pid = row["p_id"]
            if qid not in result:
                result[qid] = [pid]
            else:
                result[qid].append(pid)

        return result


class MedicalSciencesQrelDataReader(BaseHFDataQrelReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/Medical-Sciences/qrels.jsonl"
        self.name = "MedicalSciences"
        self.save_dir = os.path.join(save_path_qrel, self.name)
        self._load()


class PMCTreatmentQrelDataReader(BaseHFDataQrelReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/PMC-Treatment/qrels.jsonl"
        self.name = "PMCTreatment"
        self.save_dir = os.path.join(save_path_qrel, self.name)
        self._load()


class IIYiClinicalQrelDataReader(BaseHFDataQrelReader):
    def __init__(self):
        self.hf_path = "hf://datasets/R2MED/IIYi-Clinical/qrels.jsonl"
        self.name = "IIYiClinical"
        self.save_dir = os.path.join(save_path_qrel, self.name)
        self._load()


# Driver code, only for testing
def main():
    # data_reader = PMCTreatmentDataReader()
    # paragraphs, pids = data_reader.get_documents()
    # my_logger.info(paragraphs[0])
    # my_logger.info()
    # my_logger.info(pids[0])
    data_reader = IIYiClinicalQrelDataReader()
    print(data_reader.get_qid_to_pids())


if __name__ == "__main__":
    main()
