from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
model = AutoModel.from_pretrained("facebook/contriever-msmarco")

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    summed = token_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
    return summed / counts

# Example usage:
sentences = ["Hello world", "Another example"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
print(embeddings)