# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import (
#     Language,
#     RecursiveCharacterTextSplitter,
# )
# from langchain_core.prompts import ChatPromptTemplate
# from loader import docs



# # specify which embedding model to use
# embeddings = OllamaEmbeddings()

# # split by class and function
# separators = ['\nclass', '\ndef']
# splitter = RecursiveCharacterTextSplitter(separators=separators)

# # from_tiktoken, from_hugging_face...
# splitter = splitter.from_language(
#     language=Language.PYTHON, chunk_size=50, chunk_overlap=0
# )


# documents = splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)

# from langchain.chains.combine_documents import create_stuff_documents_chain

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)