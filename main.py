import requests
import numpy as np
import pandas as pd
from openai import OpenAI
from os import environ
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from numpy.linalg import norm

OPEN_AI_KEY = environ.get('OPEN_AI_KEY')


client = OpenAI(api_key=OPEN_AI_KEY)
####################################################################
# load documents
####################################################################
# URL of the Wikipedia page to scrape
url = "https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all the text on the page
text = soup.get_text()
text = text.replace('\n', '')

# Open a new file called 'output.txt' in write mode and store the file object in a variable
with open('output.txt', 'w', encoding='utf-8') as file:
    # Write the string to the file
    file.write(text)

# load the document
with open('./output.txt', encoding='utf-8') as f:
    text = f.read()


####################################################################
# split text
####################################################################
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.create_documents([text])

# define the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

# use the text chunks and the embeddings model to fill our vector store
db = Chroma.from_documents(texts, embeddings)

users_question = "Who was the first Prime Minister of the UK?"

# use our vector store to find similar text chunks
results = db.similarity_search(
    query=users_question,
    k=5
)

####################################################################
# build a suitable prompt and send it
####################################################################
# define the LLM you want to use
llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_KEY)

# define the prompt template
template = """
You are a chat bot who loves to help people! Given the following context sections, answer the
question using only the given context. If you are unsure and the answer is not
explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question:
{users_question}

Answer:
"""

prompt_template = PromptTemplate.from_template(template).format(
    context=results, users_question=users_question
)

print(llm.invoke(prompt_template).content)
