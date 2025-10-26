from datasets import load_dataset, load_from_disk, DatasetDict # type: ignore
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore

my_logger= my_logger.MyLogger()


class HFDatasetManager:
    def __init__(self, dataset_name_or_path: str):
        self.source = dataset_name_or_path
        self.dataset: DatasetDict = None

    def load(self, split: str | None = None, **kwargs) -> DatasetDict:
        if self.source.startswith("hf://"):
            # It's a Hugging Face file URI
            my_logger.info(f"Loading remote file from {self.source}")
            self.dataset = load_dataset("json", data_files=self.source, split=split, **kwargs)

        elif os.path.exists(self.source):
            # Load from local disk
            my_logger.info(f"Loading dataset from local path: {self.source}")
            self.dataset = load_from_disk(self.source)

        else:
            # Try to load a named dataset from the Hub
            my_logger.info(f"Loading dataset from Hugging Face Hub: {self.source}")
            self.dataset = load_dataset(self.source, split=split, **kwargs)

        return self.dataset
    
    def save(self, output_dir: str):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call `load()` first.")
        os.makedirs(output_dir, exist_ok=True)
        my_logger.info(f"Saving dataset to {output_dir} ...")
        self.dataset.save_to_disk(output_dir)
        my_logger.info("Dataset saved successfully.")
    
    def info(self):
        if self.dataset is None:
            my_logger.debug("No dataset loaded.")
            return

        if isinstance(self.dataset, DatasetDict):
            my_logger.info("Dataset splits:")
            for split, ds in self.dataset.items():
                my_logger.info(f"{split}: {len(ds)} samples, columns = {ds.column_names}", 1)
        else:
            my_logger.info(f"Single dataset with {len(self.dataset)} samples, columns = {self.dataset.column_names}")
