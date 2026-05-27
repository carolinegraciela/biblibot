import os
import torch

from dotenv import load_dotenv
load_dotenv()

# Langchain Community Modules
from langchain_groq import ChatGroq
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class LLmService():
    def __init__(self):
        self.__groq_token = os.getenv("GROQ_API")
        self.__reranker_model = os.getenv("RERANKER_MODEL")

    def generateResponse(self):
        llm = ChatGroq(
            api_key = self.__groq_token,
            temperature = 0.1,
            model_name = "llama-3.1-8b-instant",
            max_tokens = 1024,
            timeout = None,
            max_retries = 2,
        )
        return llm

    def rerankerModel(self):
        reranker = HuggingFaceCrossEncoder(
            model_name = self.__reranker_model
        )
        return reranker