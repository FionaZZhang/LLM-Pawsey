import streamlit as st
import torch
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from huggingface_hub import login

device = torch.device("cuda")

def hf_login():
  token = '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'
  login(token=token, add_to_git_credential=True)

def get_retriever():
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
  retriever = vectorstore.as_retriever()
  return retriever

def get_model():
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
  return {'tok': tokenizer, 'model': fine_tuned_model}


def handle_userinput(text, tok, llm, ret):
    retrieved_docs = ret.invoke(text)
    retrieved_page_content = retrieved_docs[0].page_content

    # Calculate the total number of words in the page content
    total_words = len(retrieved_page_content.split())

    # Calculate the starting and ending indices for the middle 100 words
    start_index = max(0, total_words // 2 - 50)  # Ensure the start index is not negative
    end_index = start_index + 100

    # Extract the middle 100 words
    retrieved = ' '.join(retrieved_page_content.split()[start_index:end_index])

    source = retrieved_docs[0].metadata['source']
    source = source.split('/')
    source = source[-1]

    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tok,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=2
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

    generated_text = sequences[0]['generated_text']
    question = st.markdown(f"<div style='background-color: #c3e3fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><span style='color: blue;'> Question   </span> <span style='color: black;'>{text}</span></div>", unsafe_allow_html=True)
    context = st.markdown(f"<div style='background-color: #e8f5ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><span style='color: purple;'><b> Context </b> {source} </span><br><span style='color: black;'> ...{retrieved}... </span></div>", unsafe_allow_html=True)

    # Extract the answer part
    answer_start = generated_text.find("Answer:") + len("Answer:")

    # Find the next occurrence of "Question:" or "Context:"
    question_start = generated_text[answer_start:].find("Question:")
    context_start = generated_text[answer_start:].find("Context:")

    # Adjust indices by adding answer_start
    question_start = question_start + answer_start if question_start != -1 else len(generated_text)
    context_start = context_start + answer_start if context_start != -1 else len(generated_text)

    # Identify the end index of the answer
    answer_end = min(question_start, context_start)
    answer = generated_text[answer_start:answer_end].strip()
    st.markdown(f"<div style='background-color: #9acef8; padding: 10px; border-radius: 5px;'><span style='color: purple;'> Answer </span><br><span style='color: black;'>{answer}</span></div>", unsafe_allow_html=True)

    return {'Question': f"{text}", 'Context': f"{retrieved}", 'Answer': f"{answer}"}

def main():
  if 'hf' not in st.session_state:
    hf_login()
    st.session_state['hf'] = True

  if 'model' not in st.session_state:
    mod_tok = get_model()
    tok = mod_tok['tok']
    mod = mod_tok['model']
    ret = get_retriever()
    st.session_state['model'] = mod
    st.session_state['token'] = tok
    st.session_state['retriever'] = ret
  
  if 'n_prompt' not in st.session_state:
    st.session_state['n_prompt'] = 0

  tok = st.session_state['token']
  mod = st.session_state['model']
  ret = st.session_state['retriever']

  st.set_page_config(page_title="Private LLM - RAG with Fine-tuned Mistral",
                       page_icon=":books:")

  st.header("Private LLM - RAG with Fine-tuned Mistral")
  st.write("Developed by Fiona")


  conversation_history = st.session_state.get('conversation_history', [])
  query = st.text_input('Enter your prompt here', key=f"{st.session_state['n_prompt']}")
  if query:
    result = handle_userinput(query, tok, mod, ret)
    conversation_history.append(result)
    st.session_state['conversation_history'] = conversation_history
    st.session_state['n_prompt'] += 1



  # Create a sidebar for conversation history
  st.sidebar.title("Conversation History")

  # Get a list of unique user questions
  user_questions = ['No Selection'] + [result['Question'] for result in conversation_history]


  # Allow the user to select a question from the sidebar
  selected_question = st.sidebar.selectbox("Select a history to display", user_questions)
  
  # Display the selected question's context and answer in the main area
  if selected_question != 'Not selected':
    for result in conversation_history:
        if result['Question'] == selected_question:
            st.sidebar.markdown(f"<div style='background-color: #c3e3fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><span style='color: blue;'> Question   </span> <span style='color: black;'>{result['Question']}</span></div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='background-color: #e8f5ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><span style='color: purple;'><b> Context </b> </span><br><span style='color: black;'> ...{result['Context']}... </span></div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='background-color: #9acef8; padding: 10px; border-radius: 5px;'><span style='color: purple;'> Answer </span><br><span style='color: black;'>{result['Answer']}</span></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
