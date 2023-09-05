from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
import os

documents_location = os.environ["DOCUMENTS_DEPOT"] + "/contoso-islands.txt"

loader = TextLoader(documents_location, encoding="utf-8")

documents = loader.load()
# text_splitter = CharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model: str = "tresnublesvectors"  # Deployment instance name for text-embedding-ada-002

vector_store_address: str = os.environ["COGNITIVE_SEARCH_ENDPOINT"]
vector_store_password: str = os.environ["COGNITIVE_SEARCH_ADMIN_KEY"]

embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
index_name: str = "langchain-contoso-islands"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

print(docs)

vector_store.add_documents(documents=docs)
