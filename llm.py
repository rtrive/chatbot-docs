from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from os import environ


OPENAI = "openai"
OLLAMA = "ollama"
HUGGINGFACE = "huggingface"

class llm:
    def __init__(self, model="llama2", temperature=0.5):
        self.model = model
        self.temperature = temperature
        
        self.llm = Ollama(model=self.model, temperature=self.temperature)

    def set_mistral(self):
        self.model = "mistral"
        self.temperature = 0.5
        self.llm = Ollama(model=self.model, temperature=self.temperature)

    def set_openai(self):
        self.llm = OpenAI(api_key=environ["OPENAI_API_KEY"])

    def invoke(self, prompt):
        results = db.similarity_search(query=prompt, k=2)
        prompt_template = PromptTemplate.from_template(self.response_template).format(
            context=results, users_question=prompt
        )
        return self.llm.invoke(prompt_template)
    
    def set_response_template(self, response_template=None):
        if response_template == None:
            self.response_template = """
            Context sections:
            {context}

            Question:
            {users_question}

            Answer:
            """
        else:
            self.response_template = response_template


db = Chroma(
    persist_directory="./db",
    embedding_function=embeddings,
    collection_name="uk_prime_minister_wikipedia_pdf",
)
