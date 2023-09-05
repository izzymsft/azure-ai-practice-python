from langchain.document_loaders import TextLoader
import os

documents_location = os.environ["DOCUMENTS_DEPOT"] + "/contoso-islands.txt"

loader = TextLoader(documents_location, encoding="utf-8")

documents = loader.load()

print(documents)
