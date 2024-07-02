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
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import glob
from splitting import parse_functions


# initial llm model
# !!! how to make model use chat history
llm = Ollama(model="mistral")
prompt_specification = PromptTemplate.from_template("""
<s>
[INST]
You are a world-class Python code documentation writer. Your task is to write specifications for Python functions in a clear and concise manner. You format these specifications in Python comments using the following structure:

\n# REQUIRES: …
\n# MODIFIES: …
\n# EFFECT: …

Here’s what each section represents:

REQUIRES: This section describes any prerequisites or conditions that must be met for the function to execute correctly. Include information such as required input parameters, preconditions, or any specific environment setup needed.

MODIFIES: Specify what aspects of the system or data structures are potentially altered or updated by the function. This could include variables that are modified, files that are written to, databases that are updated, etc.

EFFECT: Describe the overall effect or purpose of the function. This should clarify what the function accomplishes when executed, such as computations performed, data transformations, outputs generated, or side effects observed. Keep this section top-level, and give a black box idea to the user on what this function does.

Imagine a scenario where the user asks for documentation on a Python function. Your task is to understand the function's purpose and behavior, and then generate the appropriate documentation using the specified format.

For example, if the user asks you to write Python function documentation on this following function:

def sort_integer_list(numbers):
    sorted_numbers = sorted(numbers)
    return sorted_numbers

then you should write the appropriate function documentation in this format:

\n# REQUIRES: A list of integers `nums` to sort.
\n# MODIFIES: The order of elements in the input list `nums`.
\n# EFFECT: Sorts the list `nums` in ascending order using a stable sorting algorithm.

Ensure the documentation is clear and concise, focusing on the essential aspects of the function’s behavior and requirements. Use precise language and terminology commonly understood in the context of Python programming and software documentation. Only write comments, do not write code.
[/INST]
</s>
[INST]Write Python function documentation for the function {input}, given its detailed implementation {code}.[/INST]
"""
)



# 
prompt_explainer = PromptTemplate.from_template("""
<s>
[INST]
You are tasked with explaining a coding function or class in detail. Your role is to provide a comprehensive explanation that covers the purpose, functionality, usage, parameters, and any other relevant details.

Here are your instructions:
Function/Class Description: Provide a clear and concise description of the function or class.
Function/Class Purpose: Explain the primary objective or goal that the function/class aims to achieve.
Parameters: List and describe all input parameters and their roles in the function/class.
Return Value (if applicable): Explain what the function returns or the effects of the class instantiation.
Usage Example: Provide a practical example demonstrating how the function/class is used in context.
Special Cases or Edge Cases: If applicable, describe any special or edge cases that the function/class handles differently.
Algorithm or Methodology: Describe the underlying algorithm, methodology, runtime, or approach used by the function/class to accomplish its task.
Dependencies or Requirements: List any external dependencies, libraries, or environmental prerequisites needed for the function/class to operate correctly.
Format:
Use clear and precise language appropriate for technical documentation.
Ensure that your explanation is thorough but accessible to a technically literate audience.
Do not include actual code; focus solely on explaining the function/class and its behavior.

Now here is an example:

Imagine a scenario where a user asks for an explanation of a Python function or class. Your task is to understand the function/class's purpose and behavior and then provide a detailed explanation based on the specified format.

For instance, the user asks you to explain this function:

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

Your explanation should cover all aspects outlined in the instructions, providing a comprehensive understanding of how the factorial function works, its purpose, parameters, and usage.

Tailor your explanation to the specific function or class requested by the user.
Use technical terminology relevant to the programming language and concepts involved.
Aim for clarity and completeness in your explanation, addressing both basic functionality and any advanced features or considerations.
[/INST]
The factorial function calculates the factorial of a non-negative integer n. Factorial is defined as the product of all positive integers less than or equal to n. The function begins with a base case check: if n equals 0, it returns 1. This handles the termination condition of the recursive function, ensuring that factorial(0) returns 1, a fundamental rule in factorial mathematics. For any other positive integer n, the function recursively calls itself with n-1, multiplying n by the result of factorial(n-1). This recursion continues until reaching the base case, at which point it returns up the call stack, computing the factorial by multiplying successive integers. The function’s recursive nature efficiently computes factorials but requires careful handling of large values to avoid stack overflow in some programming environments.</s>
</s>
[INST]Explain {input} to me in detail, given its implementation in code {code} and its local callees with respective implementations {callees} [/INST]
"""
)

output_parser = StrOutputParser()

def call_llm(func, mode):
    print(func)
    chain = get_llm_chain(mode)
    print(chain.invoke({"input": func[0], "code": func[3], "callees": func[4]}))
    
def get_llm_chain(mode):
    chain = None
    if mode == "explanation":
        chain = prompt_explainer | llm | output_parser
    if mode == "specification":
        chain = prompt_specification | llm | output_parser
    return chain

def process_all_python_files(directory):
    python_files = glob.glob(os.path.join(directory, '*.py'), recursive=True) # !!!
    for file in python_files:
        functions, source_code = parse_functions(file)
        print(f"Functions in {file}:")
        for function in functions:
            print(f" - {function[0]} (lines {function[1]}-{function[2]})")
            call_llm(function, 'specification')
            
process_all_python_files('.')

# loader = DirectoryLoader('../', glob="**/*.py", show_progress=True, loader_cls=PythonLoader, recursive=True) # loads python source codes recursively from directory. Maybe default to TextLoader
# # use other json + html + pdf + markdown loaders?

# docs = loader.load()

# # check how many files, do something with it
# print(len(docs))


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

# # document_chain = llm | prompt
# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = vector.as_retriever()

# # retriever_chain = llm | prompt | retriever
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # can only answer single question, good for function specific questions
# response = retrieval_chain.invoke({"input": "what is this function Main()"})
# print(response["answer"])



# # history aware
# # First we need a prompt that we can pass into an LLM to generate this search query

# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
# ])

# # retriever_chain = llm | retriever | prompt?
# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# chat_history = [HumanMessage(content="function name"), AIMessage(content="main()")]
# retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])

# # new document chain?
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# # end to end
# retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell its implementation"
# })