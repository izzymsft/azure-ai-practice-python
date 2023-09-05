from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os

documents_location = os.environ["DOCUMENTS_DEPOT"] + "/contoso-islands.txt"

loader = TextLoader(documents_location, encoding="utf-8")

documents = loader.load()
# text_splitter = CharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display the documents
print(docs)

# Show number of elements in the array
print(len(docs))


