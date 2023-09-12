from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
import os

llm_name = "tresnublesgpt35"
model_name: str = "tresnublesvectors"  # The Deployment instance name for text-embedding-ada-002

vector_store_address: str = os.environ["COGNITIVE_SEARCH_ENDPOINT"]
vector_store_password: str = os.environ["COGNITIVE_SEARCH_ADMIN_KEY"]

embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model_name, chunk_size=1)
index_name: str = "langchain-contoso-islands"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

llm = ChatOpenAI(model_name=llm_name, temperature=0, model_kwargs={"deployment_id": llm_name})

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Be very excited in your response! 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Where does the president and vice president live?"

# Perform a hybrid search
docs = vector_store.similarity_search(query=question, k=3, search_type="hybrid")

prompt_results = qa_chain({"query": question})

print(prompt_results["query"])

print(prompt_results["result"])

# print(prompt_results["source_documents"])
