
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import getpass
import os
from langchain_cohere import CohereEmbeddings
import lancedb

loader = DirectoryLoader('.', glob="**/*.py", show_progress=True, loader_cls=PythonLoader, recursive=False) # loads python source codes recursively from directory. Maybe default to TextLoader
# use other json + html + pdf + markdown loaders?

docs = loader.load()
llm = Ollama(model="mistral")
# specify which embedding model to use
# embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
print("hi")
# split by class and function
separators = ['\ndef', '\n\tdef']
splitter = RecursiveCharacterTextSplitter(
    separators=separators, chunk_overlap=0
)


documents = splitter.split_documents(docs)
print(documents)
print(len(documents))
print("hi")
# embeddings = embeddings_model.embed_documents(documents)
print("hi")
from langchain.chains.combine_documents import create_stuff_documents_chain
print("hi")
db = lancedb.connect("/tmp/lancedb")
db = LanceDB.from_documents(documents, CohereEmbeddings())
print("hi")
# retriever_chain = llm | prompt | retriever
query = "call_llm("
hi = db.similarity_search(query)
# print(hi[])
# print(len(hi))

