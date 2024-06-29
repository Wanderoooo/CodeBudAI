# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import PythonLoader

# loader = DirectoryLoader('../', glob="**/*.py", show_progress=True, show_progress=True, loader_cls=PythonLoader, recursive=True) # loads python source codes recursively from directory. Maybe default to TextLoader
# # use other json + html + pdf + markdown loaders?

# docs = loader.load()

# # check how many files, do something with it
# print(len(docs))