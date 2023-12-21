# Device: CPU/GPU
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mistral 7B
# Imports
from ctransformers import AutoModelForCausalLM

# Create Mistral model
mistral_model = AutoModelForCausalLM.from_pretrained("your_mistral_model").to(device)

# Test the Mistral model
print("\nMistral 7B:")
print(mistral_model("Can you tell me a joke?"))

# BloomZ 1b
# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BloomTokenizerFast, BloomForCausalLM

# Load the BloomZ model
param = "bloomz-1b1"
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/" + param)
model = BloomForCausalLM.from_pretrained("bigscience/" + param).to(device)

# Test the BloomZ model
inputs = tokenizer.encode("Can you tell me a joke?", return_tensors="pt").to(device)
outputs = model.generate(inputs)
decoded_output = tokenizer.decode(outputs[0])

print("\nBloomZ 1b:")
print(decoded_output)

# Llama 2
# Imports
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Load the Llama model
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
)

# See the number of layers in GPU
gpu_layers = lcpp_llm.params.n_gpu_layers
print(f"\nLlama 2 GPU Layers: {gpu_layers}")

# Test the Llama model
response = lcpp_llm(prompt="Can you tell me a joke?", max_tokens=256, temperature=0.5, top_p=0.95,
                    repeat_penalty=1.2, top_k=150,
                    echo=True)

llama_output = response['choices'][0]['text']

print("\nLlama 2 Output:")
print(llama_output)
