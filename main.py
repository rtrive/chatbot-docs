import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from os import environ


OPEN_AI_KEY = environ.get("OPEN_AI_KEY")
environ["TOKENIZERS_PARALLELISM"] = "false"

db = Chroma(
    persist_directory="./db",
    embedding_function=OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY),
    collection_name="uk_prime_minister_wikipedia",
)

users_question = "Who is the Rishi Sunak?"

# use our vector store to find similar text chunks
results = db.similarity_search(query=users_question, k=5)

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
