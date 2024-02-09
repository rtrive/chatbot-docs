import chromadb
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from os import environ


environ["TOKENIZERS_PARALLELISM"] = "false"

OPEN_AI_KEY = environ.get("OPEN_AI_KEY")


####################################################################
# load documents
####################################################################
# URL of the Wikipedia page to scrape
url = "https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom"

loaders = WebBaseLoader(url)
data = loaders.load()


####################################################################
# split text
####################################################################
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # use the text chunks and the embeddings model to fill our vector store
client = chromadb.PersistentClient(path="./db")
db = Chroma(client=client)

collection = client.get_or_create_collection("uk_prime_minister_wikipedia")
db.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name=collection.name,
    client=client,
)
