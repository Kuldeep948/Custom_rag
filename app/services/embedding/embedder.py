"""
Embedding service.

Providers
---------
sentence_transformer  — local, free, no API key (default)
huggingface           — alias for sentence_transformer
gemini                — Google Gemini text-embedding-004 (requires GEMINI_API_KEY)

Configure via app/config.yaml → embedding.provider
or the EMBEDDING_PROVIDER environment variable.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Common interface every embedding provider must implement."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string. Returns one float vector."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


# ── Sentence Transformers — local, no API key ─────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding using sentence-transformers.
    No API key required. Vectors are L2-normalised for cosine similarity.

    Recommended models
    ------------------
    Model                                   Dims   Notes
    all-MiniLM-L6-v2                         384   best default (fast)
    all-MiniLM-L12-v2                        384   slightly better quality
    all-mpnet-base-v2                        768   highest quality
    multi-qa-MiniLM-L6-cos-v1               384   tuned for Q&A
    paraphrase-multilingual-MiniLM-L12-v2   384   50+ languages
    """

    _KNOWN_DIMS = {
        "all-MiniLM-L6-v2":                       384,
        "all-MiniLM-L12-v2":                      384,
        "all-mpnet-base-v2":                      768,
        "multi-qa-MiniLM-L6-cos-v1":             384,
        "multi-qa-mpnet-base-dot-v1":            768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "all-distilroberta-v1":                   768,
        "msmarco-distilbert-base-v4":             768,
    }

    def __init__(
        self,
        model: str = None,
        device: str = None,
        normalize: bool = True,
        batch_size: int = None,
    ):
        raw = model or settings.embedding_model
        self._model_name = raw.replace("sentence-transformers/", "")
        self._full_model_name = raw
        self._device = device or settings.st_device
        self._normalize = normalize
        self._batch_size = batch_size or settings.st_batch_size
        self._model = None  # lazy-loaded on first use

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_sentence_transformer",
                        model=self._full_model_name, device=self._device)
            self._model = SentenceTransformer(self._full_model_name, device=self._device)
            logger.info("sentence_transformer_ready",
                        model=self._model_name,
                        dimension=self._model.get_sentence_embedding_dimension(),
                        device=self._device)
        return self._model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in a thread-pool so the async event loop is not blocked."""
        if not texts:
            return []
        model = self._get_model()
        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()

        embeddings = await loop.run_in_executor(None, _encode)
        logger.debug("sentence_transformer_encoded",
                     count=len(texts), model=self._model_name)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        return (await self.embed_texts([query]))[0]

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        short = self._model_name.split("/")[-1]
        return self._KNOWN_DIMS.get(short, settings.embedding_dimension)

    @property
    def model_name(self) -> str:
        return self._full_model_name


# ── Gemini Embeddings — requires GEMINI_API_KEY ───────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    """
    Google Gemini embedding provider (models/text-embedding-004, 768 dims).
    Requires GEMINI_API_KEY.
    """

    BATCH_SIZE = 100

    def __init__(self, model: str = None, api_key: str = None):
        self._model = model or settings.embedding_model or "models/text-embedding-004"
        self._api_key = api_key or settings.gemini_api_key
        self._configured = False

    def _configure(self):
        if not self._configured:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._configured = True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._configure()
        import google.generativeai as genai
        loop = asyncio.get_event_loop()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]

            def _embed_batch(b=batch):
                return [
                    genai.embed_content(
                        model=self._model,
                        content=t,
                        task_type="retrieval_document",
                    )["embedding"]
                    for t in b
                ]

            all_embeddings.extend(await loop.run_in_executor(None, _embed_batch))

        logger.debug("gemini_embeddings_generated", count=len(texts), model=self._model)
        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        self._configure()
        import google.generativeai as genai
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=self._model,
                content=query,
                task_type="retrieval_query",
            )["embedding"],
        )

    @property
    def dimension(self) -> int:
        return 768  # text-embedding-004

    @property
    def model_name(self) -> str:
        return self._model


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder() -> BaseEmbedder:
    """
    Return the configured embedding provider.

    sentence_transformer / huggingface  →  SentenceTransformerEmbedder (default)
    gemini                              →  GeminiEmbedder (requires GEMINI_API_KEY)
    """
    provider = settings.embedding_provider.lower()

    if provider in ("sentence_transformer", "huggingface"):
        return SentenceTransformerEmbedder()

    if provider == "gemini":
        if not settings.gemini_api_key:
            logger.warning("gemini_key_missing", fallback="sentence_transformer")
            return SentenceTransformerEmbedder()
        return GeminiEmbedder()

    raise ValueError(
        f"Unknown embedding_provider '{provider}'. "
        "Valid options: sentence_transformer, huggingface, gemini"
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_embedder_instance: Optional[BaseEmbedder] = None


def get_embedder_singleton() -> BaseEmbedder:
    """Return the global embedder instance (created once, reused everywhere)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = get_embedder()
    return _embedder_instance


def reset_embedder_singleton() -> None:
    """Reset the singleton — used in tests to swap providers."""
    global _embedder_instance
    _embedder_instance = None
