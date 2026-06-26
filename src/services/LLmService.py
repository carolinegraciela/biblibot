import os
import torch

from dotenv import load_dotenv
load_dotenv()

# Langchain Community Modules
from langchain_groq import ChatGroq
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_models import ChatOllama

class LLmService():
    def __init__(self):
        self.__groq_token_1 = os.getenv("GROQ_API_1")
        self.__groq_token_2 = os.getenv("GROQ_API_2")
        self.__groq_token_3 = os.getenv("GROQ_API_3")
        self.__groq_token_4 = os.getenv("GROQ_API_4")        

        # self.ip_vps = "187.77.119.65"
        self.__reranker_model = os.getenv("RERANKER_MODEL")

    def generateResponse(self):
        llm_1 = ChatGroq(
            api_key = self.__groq_token_1,
            temperature = 0.1,
            model_name = "llama-3.1-8b-instant",
            max_tokens = 1024,
            timeout = None,
            max_retries = 0,
        )

        llm_2 = ChatGroq(
            api_key = self.__groq_token_2,
            temperature = 0.1,
            model_name = "llama-3.1-8b-instant",
            max_tokens = 1024,
            timeout = None,
            max_retries = 0,
        )

        llm_3 = ChatGroq(
            api_key = self.__groq_token_3,
            temperature = 0.1,
            model_name = "llama-3.1-8b-instant",
            max_tokens = 1024,
            timeout = None,
            max_retries = 0,
        )

        llm_4 = ChatGroq(
            api_key = self.__groq_token_4,
            temperature = 0.1,
            model_name = "llama-3.1-8b-instant",
            max_tokens = 1024,
            timeout = None,
            max_retries = 0,
        )

        robust_llm = llm_1.with_fallbacks([llm_2, llm_3, llm_4])
        return robust_llm

    def rerankerModel(self):
        reranker = HuggingFaceCrossEncoder(
            model_name = self.__reranker_model
        )
        return reranker