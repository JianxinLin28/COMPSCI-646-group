# COMPSCI-646-group
A Study of Hard Negative Samples on Reasoning-aware Retrieval Models

## 📖Guide: generate negative samples using LLM
This guide will teach you how to generate negative samples using LLM.
* Step 1: create a new file '.env' under `negative_sample` folder. Enter
```
OPENAI_API_KEY = [The key]
```
Replace [The key] with an openai api key. You can use the one provided by Jianming. It's under our Notion page -> Misc. 
* Step 2: from `negative_sample`, run
```
pip install -r requirements.txt
```
* Step 3: from `negative_sample`, run
```
python supplement_negative_passage.py --input_file data/sample.jsonl
```
You should see a newly generated file called `sample_generated_negative.jsonl`. That contains a new column `neg`, which is the negative sample.

