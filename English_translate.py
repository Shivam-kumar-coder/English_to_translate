import streamlit as st
from transformers import MarianMTModel,MarianTokenizer

@st.cache_resouce
h_tokenizer= MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
h_model=MaranMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

st.title(" Language Translater 🚀")
i=st.chat_input("enter your Text")
l=st.selectbox(" Select Your Language Convert ")
if i:
  t=h_tokenizer(i,return_tensor='pt',padding=True)
  m=h_model.generate(**t)
  d=h_tokenizer.decode(m,skip_special_token=True)
  st.write(d)
