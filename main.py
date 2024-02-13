from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from os import environ


OPENAI = "openai"
OLLAMA = "ollama"
HUGGINGFACE = "huggingface"
OPENAI_API_KEY = environ["OPENAI_API_KEY"]


def set_embeddings(type, model=None):
    if type == OLLAMA:
        return OllamaEmbeddings(model=model)
    elif type == OPENAI:
        return OpenAIEmbeddings()
    elif type == HUGGINGFACE:
        model_kwargs = {"device": "cpu", "trust_remote_code": False}
        local_model_path = "./embeddings/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)
    pass


environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = set_embeddings(type=HUGGINGFACE)


db = Chroma(
    persist_directory="./db",
    embedding_function=embeddings,
    collection_name="uk_prime_minister_wikipedia_pdf",
)

users_question = "Who is the first Prime Minister of the UK?"

# use our vector store to find similar text chunks
results = db.similarity_search(query=users_question, k=2)

####################################################################
# build a suitable prompt and send it
####################################################################
# define the LLM you want to use
llm = Ollama(model="llama2", temperature=0.5)


# define the prompt template
template = """

Context sections:
{context}

Question:
{users_question}

Answer:
"""

prompt_template = PromptTemplate.from_template(template).format(
    context=results, users_question=users_question
)

print(llm.invoke(prompt_template))
