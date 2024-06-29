from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# initial llm model
llm = Ollama(model="mistral")

# 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical specification writer. Write all specifications in the format: # REQUIRES: ...# MODIFIES:..., # EFFECT: ..."),
    ("user", "{input}")
])

# parse message to string
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# chain.invoke({"input": "write documentation on {function_name}"})

loader = DirectoryLoader('../', glob="**/*.py", show_progress=True, show_progress=True, loader_cls=PythonLoader, recursive=True) # loads python source codes recursively from directory. Maybe default to TextLoader
# use other json + html + pdf + markdown loaders?

docs = loader.load()

# check how many files, do something with it
print(len(docs))


# specify which embedding model to use
embeddings = OllamaEmbeddings()

# split by class and function
separators = ['\nclass', '\ndef']
splitter = RecursiveCharacterTextSplitter(separators=separators)

# from_tiktoken, from_hugging_face...
splitter = splitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)


documents = splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# document_chain = llm | prompt
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()

# retriever_chain = llm | prompt | retriever
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# can only answer single question, good for function specific questions
response = retrieval_chain.invoke({"input": "what is this function Main()"})
print(response["answer"])



# history aware
# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

# retriever_chain = llm | retriever | prompt?
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
