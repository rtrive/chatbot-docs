class embedding:
    def __init__(self, type, model=None):
        self.type = type
        self.model = model


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