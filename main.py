import streamlit as st
import torch
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel


print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated
device = torch.device("mps")

# Log in Huggingface
token = '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'
from huggingface_hub import login
login(token=token, add_to_git_credential=True)

# Initializing Pinecone Vector DB
pinecone.init(
    api_key='5cfb7c53-9206-4efe-b27c-beb0f61ef496',
    environment='gcp-starter'
)

# Pinecone Vector DB index name
index_name = 'csiro-vector'
index = pinecone.Index(index_name)

# Index using vector store just built
text_field = "text"
embeddings = HuggingFaceEmbeddings()

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)

# Load the tokenizer, adjust configuration if needed
model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# Load the fine-tuned model with its trained weights
fine_tuned_model = PeftModel.from_pretrained(model, 'fionazhang/mistral_7b_environment')


"""
# Private LLM - RAG with Fine-tuned Mistral
"""
query = st.text_input('Enter your prompt here', 'E.g. How does CSIRO respond to climate change?')

# Get RAG answer
def get_ans(text, tok, llm, ret):

  retrieved_docs = ret.invoke(query)
  retrieved = retrieved_docs[0].page_content[100:500]

  pipe = pipeline(
      "text-generation",
      model=llm,
      tokenizer = tok,
      torch_dtype=torch.bfloat16,
      device_map="auto",
      pad_token_id = 2
  )

  sequences = pipe(
      f"Context: {retrieved} \n Question: {text} \n Answer: ",
      do_sample=True,
      max_new_tokens=100,
      temperature=0.7,
      top_k=50,
      top_p=0.95,
      num_return_sequences=1,
  )
  return sequences[0]['generated_text']

retriever = vectorstore.as_retriever()
st.write(get_ans(query, tokenizer, model, retriever))
