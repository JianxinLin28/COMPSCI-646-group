# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="ncbi/MedCPT-Query-Encoder")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")