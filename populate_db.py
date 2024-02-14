import chromadb

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores.chroma import Chroma
from os import environ

OPENAI = "openai"
OLLAMA = "ollama"
HUGGINGFACE = "huggingface"
OPEN_AI_KEY = environ["OPENAI_API_KEY"]


def set_embeddings(type, model=None):
    if type == OLLAMA:
        return OllamaEmbeddings(model=model)
    elif type == OPENAI:
        return OpenAIEmbeddings()
    elif type == HUGGINGFACE:
        model_kwargs = {"device": "cpu", "trust_remote_code": False}
        local_model_path = "./embeddings/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=local_model_path, model_kwargs=model_kwargs
        )
    pass


environ["TOKENIZERS_PARALLELISM"] = "false"


####################################################################
# load documents
####################################################################
# URL of the Wikipedia page to scrape
pdf_path = "pdfs/uk.pdf"

loaders = PyPDFLoader(pdf_path)
pages = loaders.load()


####################################################################
# split text
####################################################################
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

texts = text_splitter.split_documents(pages)

embeddings = set_embeddings(type=HUGGINGFACE)

# # use the text chunks and the embeddings model to fill our vector store
client = chromadb.PersistentClient(path="./db")
db = Chroma(client=client)

collection = client.get_or_create_collection("uk_prime_minister_wikipedia_pdf")
db.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name=collection.name,
    client=client,
)
