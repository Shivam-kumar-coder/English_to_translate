import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
@st.cache_resource
def process():
  tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
  model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
  return model 
  return tokenize

st.title(" Language Translater 🚀")
i=st.chat_input("enter your Text")
l=st.selectbox(" Select Your Language Convert ",("hindi"),)
if i:
  model,tokenize=process()
  t=tokenize(i,return_tensor='pt',padding=True)
  m=model.generate(**t)
  d=tokenize.decode(m,skip_special_tokens=True)
  st.write(d)
else:
  st.write("please enter your text")
