This page records the parts of code that is written / not written by our group.

# Synthetic data generation
Code written by Jianxin Lin:

* `synthetic_data_generation/MyCode/data_reader.py`
    * Everything: Download, cache, load datasets from hugging face.
* `synthetic_data_generation/MyCode/hard_query_gen.py`
    * doc2query (modified to target specific queries): The original function processes random documents from the entire corpus. I have made it target only specific documents.
    * get_pairs_used_pid: Helper function for making doc2query targeting specific documents.
    * get_generated_qids: Helper function for making doc2query targeting specific documents.
    * add_to_record: Helper function for making doc2query targeting specific documents.
    * get_pids: Helper function for making doc2query targeting specific documents.
    * main: The driver code. Added new logic to load datasets and make doc2query targeting specific documents.
* `synthetic_data_generation/MyCode/hf_dataset_manager.py`
    * Everything: Custom hugging face dataset manager.
* `synthetic_data_generation/MyCode/hq_to_hard_neg_doc.py`
    * main: The script is mostly the same as the original. The only change to it is to specify input & output paths.
* `synthetic_data_generation/MyCode/qrel_maker.py`
    * Everything: Generates new hard passage id to hard passage pairs that are going to be added to the new corpus. This script has a bad name due to an early-stage misunderstanding. We decided to keep the name because everyone has gotten used to it.
* `synthetic_data_generation/MyUtil/my_logger.py`
    * Everything: A singleton class for logging.

Code written by ReasonIR Researchers:

* Everything else under `synthetic_data_generation`. Some of them might have been slightly modified for debugging purpose.
