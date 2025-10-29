from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, provider: str, model_name: str, chunk_size: int = 512, chunk_overlap: int = 50):
        self.provider = provider.lower()
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed_model = self._init_embed_model()
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        Settings.embed_model = self.embed_model
        Settings.text_splitter = self.text_splitter

        logger.info(f"Config initialized: {provider} / {model_name}")

    def _init_embed_model(self):
        try:
            if self.provider == "huggingface":
                hf_token = os.getenv("HF_TOKEN")

                return HuggingFaceEmbedding(
                    model_name=self.model_name,
                    trust_remote_code=True,
                    embed_batch_size=32,
                    cache_folder="./models_cache"
                )

            elif self.provider == "ollama":
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                return OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=ollama_url,
                )

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
