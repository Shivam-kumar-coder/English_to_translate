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
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-pa')
    return model,tokenize
  def tamil(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ta')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ta')
    return model,tokenize
  def urdu(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ur')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ur')
    return model,tokenize
  def marathi(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-mr')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-mr')
    return model,tokenize
  def gujrati(self):
    tokenize = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-gu')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-gu')
    return model,tokenize

st.title(" Language Translater 🚀")
i=st.chat_input("enter your Text")
l=st.selectbox(" Select Your Language Convert ",["Hindi","Urdu","Punjabi","Gujrati","Tamil","Marathi"])
if i:
  if l=="Urdu":
    with st.spinner("Convert Into Urdu....."):
      p=trans()
      model,tokeniz=p.urdu()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)
  elif l=="Hindi":
    with st.spinner("Convert Into Hindi....."):
      p=trans()
      model,tokeniz=p.hindi()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)
  elif l=="Gujrati":
    with st.spinner("Convert Into Gujrati...."):
      p=trans()
      model,tokeniz=p.gujrati()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)
  elif l=="Tamil":
    with st.spinner("Convert Into Tamil...."):
      p=trans()
      model,tokeniz=p.tamil()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)
  elif l=="Marathi":
    with st.spinner("Convert Into Marathi....."):
      p=trans()
      model,tokeniz=p.marathi()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)
  elif l=="Punjabi":
    with st.spinner("convert into Punjabi ...."):
      p=trans()
      model,tokeniz=p.punjabi()
      t=tokeniz(i,return_tensors='pt',padding=True)
      m=model.generate(**t)
      d=tokeniz.decode(m[0],skip_special_tokens=True)
      st.success("Converted Succesfully")
      st.write(d)  
else:
  st.info("Please Enter Your Text")
