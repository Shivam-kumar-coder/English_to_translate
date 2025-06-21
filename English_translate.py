import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
@st.cache_resource
class trans:
  def hindi(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
    return model, tokenize
  def punjabi(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-pa')
    model = "Helsinki-NLP/opus-mt-en-pa"
    return model,tokenize
  def tamil(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ta')
    model_name = "Helsinki-NLP/opus-mt-en-ta"
    return model,tokenize
  def urdu(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ur')
    model = "Helsinki-NLP/opus-mt-en-ur"
    return model,tokenize
  def marathi(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-mr')
    model = "Helsinki-NLP/opus-mt-en-mr"
    return model,tokenize
  def gujrati(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-gu')
    model = "Helsinki-NLP/opus-mt-en-gu"
    return model,tokenize

st.title(" Language Translater 🚀")
i=st.chat_input("enter your Text")
l=st.selectbox(" Select Your Language Convert ",["hindi","Urdu","Gujrati","Tamil","Marathi"])
if i:
  if l=="Urdu":
    p=trans()
    mode,tokeniz=p.urdu()
    t=tokeniz(i,return_tensors='pt',padding=True)
    m=mode.generate(**t)
    d=tokeniz.decode(m[0],skip_special_tokens=True)
    st.write(d)
else:
  st.write("please enter your text")
