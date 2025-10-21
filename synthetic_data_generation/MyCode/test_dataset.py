import pandas as pd
from huggingface_hub import login # type: ignore
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))

import my_logger # type: ignore

my_logger= my_logger.MyLogger()

# login()

df = pd.read_json("hf://datasets/R2MED/Medical-Sciences/corpus.jsonl", lines=True)

for item in df["text"][:5]:
    my_logger.info(item)
    my_logger.info()

