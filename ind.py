from langchain.document_loaders import PyPDFLoader
import openai
import streamlit as st
from dotenv import load_dotenv
import os

# 1loading--load pdf file 
loader=PyPDFLoader('agrQA.pdf')
# 2splitting--Break document into split of specific size
page_contant=loader.load_and_split()
print(len(page_contant),page_contant)
#3Retrival: the app retrives split from storage that matched with user query 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


load_dotenv()
# openai_api_key=os.getenv("api_secret")
openai.api_key=os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings()
db=FAISS.from_documents(page_contant,embeddings)
#save in to local
db.save_local("faiss_index")
