import json
import sys
import os
from typing import List
from negative_passage_gen import NegativePassageGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MyUtil')))
import my_logger # type: ignore


my_logger= my_logger.MyLogger()

# ================================================
data_path = "./outputs/synthetic_questions/all_docs_train_data/hq_gen/gpt-4o"
output_path = "./outputs/negative_passages"
num_workers: int = None
# ================================================


def get_jsonl_files(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.jsonl') and os.path.isfile(os.path.join(dir_path, f))
    ]


def format_data(data: List[dict]) -> List[dict]:
    result = []
    count = 0
    for obj in data:
        new_obj = {}
        new_obj["query"] = [obj["query"], "", f"q{count}"]
        new_obj["pos"] = [["", obj["pos"][0]]]
        count += 1
        result.append(new_obj)
    return result


def is_already_generated(jsonl_file):
    # Check if it's in output path
    file_name = os.path.basename(jsonl_file)
    supposed_save_path = os.path.join(output_path, file_name)
    return os.path.exists(supposed_save_path)


def hq_to_hard_neg_doc():
    jsonl_files = get_jsonl_files(data_path)

    for jsonl_file in jsonl_files:
        # If jsonl file is already generated, skip
        if is_already_generated(jsonl_file):
            my_logger.warn(f"{jsonl_file} already generated. Skip.")
            continue

        my_logger.info(jsonl_file)

        data = []
        with open(jsonl_file, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
        
        data = format_data(data)

        file_name = os.path.basename(jsonl_file)

        if num_workers is not None:
            for i in range(num_workers):
                output_name = file_name.replace(".jsonl", f"_id{i}.jsonl")
                output_file = os.path.join(output_path, output_name)
                neg_gen = NegativePassageGenerator(data, output_file, num_workers, i)
                neg_gen.generate()
        else:
            output_file = os.path.join(output_path, file_name)
            neg_gen = NegativePassageGenerator(data, output_file)
            neg_gen.generate()


def main():
    hq_to_hard_neg_doc()


if __name__ == "__main__":
    main()
