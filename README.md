# COMPSCI-646-group
A Study of Hard Negative Samples on Reasoning-aware Retrieval Models

## 📘Guide 1: generate negative samples using LLM (Decrypted)
⚠️ **Decrypted, follow Guide 2 instead**

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

## 📖Guide 2: generate negative samples using LLM 
⚠️ I strongly recommend you to install everything in a dedicated python environment. (>= python 3.10)

This is the newer guide that teaches you how to generate better negative samples using LLM.
* Step 1: create a new file '.env' under `synthetic_data_generation` folder. Enter
```
OPENAI_API_KEY = [The key]
```
Replace [The key] with an openai api key. You can use the one provided by Jianming. It's under our Notion page -> Misc. 

* Step 2: from `synthetic_data_generation`, run
```
pip install -r requirements.txt
```

* Step 3: 

if you are using Linux, run:
```
bash setup_java_linux.sh
```
if you are using Windows, run:
```
bash setup_java_winos.sh
```
