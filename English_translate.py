import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
@st.cache_resource
def process():
  global tokenize
  global model
  tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
  model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

st.title(" Language Translater 🚀")
i=st.chat_input("enter your Text")
l=st.selectbox(" Select Your Language Convert ")
if i:
  process()
  t=tokenize(i,return_tensor='pt',padding=True)
  m=model.generate(**t)
  d=tokenize.decode(m,skip_special_token=True)
  st.write(d)
