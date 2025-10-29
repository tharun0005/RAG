import os
import sys
import logging
from typing import List, Dict, Any, Optional
from langfuse.decorators import observe, langfuse_context

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = logging.getLogger(__name__)

try:
    from ingestion.config import Config
    from ingestion.vector_store_manager import vector_store_manager

    IMPORT_SUCCESS = True
    logger.info("Successfully imported ingestion modules")
except ImportError as e:
    logger.error(f"Failed to import ingestion modules: {e}")
    IMPORT_SUCCESS = False

try:
    from langfuse import Langfuse

    LANGFUSE_ENABLED = True
    logger.info("LangFuse available for Retriever")
except ImportError:
    LANGFUSE_ENABLED = False
    logger.warning("LangFuse not available - tracing disabled")


class Retriever:
    def __init__(
            self,
            store_id: str,
            store_name: str,
            provider: str,
            model_name: str,
            top_k: int = 5,
            score_threshold: float = 0.2,
            include_source: bool = True,
    ):
        """
        Initialize retriever with embedding model and vector store info

        Args:
            store_id: Unique identifier for the vector store
            store_name: Name of the vector store
            provider: Embedding provider ("huggingface" or "ollama")
            model_name: Name of the embedding model
            top_k: Number of top results to retrieve
            score_threshold: Minimum similarity score (0-1, will be converted if needed)
            include_source: Whether to include source information
        """
        if not IMPORT_SUCCESS:
            raise ImportError("Ingestion modules not available. Check Python path and imports.")

        self.store_id = store_id
        self.store_name = store_name
        self.config = Config(provider, model_name)
        self.top_k = top_k
        self.score_threshold = self._normalize_threshold(score_threshold)
        self.include_source = include_source

        logger.info(f"Initialized Retriever for store: {store_name} (ID: {store_id})")
        logger.info(f"   Provider: {provider}, Model: {model_name}")
        logger.info(f"   Parameters: top_k={top_k}, threshold={self.score_threshold:.3f}")

    def _normalize_threshold(self, threshold: float) -> float:
        """
        Normalize threshold to work with cosine similarity (-1 to 1)

        If threshold is between 0-1, convert it to cosine similarity range
        by mapping 0->-1, 0.5->0, 1->1
        """
        if threshold < 0:
            return threshold

        normalized = (threshold * 2) - 1
        logger.debug(f"Normalized threshold {threshold} -> {normalized}")
        return normalized

    @observe(name="retrieve_context", capture_input=True, capture_output=True)
    async def retrieve_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant context from vector store for the given query

        Args:
            query: User query string

        Returns:
            Dictionary with context and sources
        """
        try:
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_trace(
                    tags=["retrieval", "vectorstore", self.store_name],
                    metadata={
                        "store_id": self.store_id,
                        "store_name": self.store_name,
                        "provider": self.config.provider,
                        "model": self.config.model_name,
                        "top_k": self.top_k,
                        "score_threshold": self.score_threshold
                    }
                )

                langfuse_context.update_current_observation(
                    input={
                        "query": query,
                        "query_length": len(query) if query else 0
                    }
                )

            if not query or not query.strip():
                logger.warning("Empty query received")
                return self._empty_response()

            query = query.strip()
            logger.info(f"Retrieving context for query: '{query[:50]}...' from store: {self.store_name}")

            logger.debug("Getting vector store index...")
            index = vector_store_manager.get_vector_store_index(
                self.store_id,
                self.store_name,
                self.config.embed_model
            )

            if not index:
                logger.error(f"Vector store index not found: {self.store_name} (ID: {self.store_id})")

                if LANGFUSE_ENABLED:
                    langfuse_context.update_current_observation(
                        level="ERROR",
                        status_message="Vector store index not found"
                    )

                return self._empty_response()

            retrieval_top_k = max(self.top_k * 2, 10)
            logger.debug(f"Using retriever with top_k={retrieval_top_k} (will filter to {self.top_k})")
            retriever = index.as_retriever(similarity_top_k=retrieval_top_k)

            logger.debug("Retrieving similar documents...")
            nodes = retriever.retrieve(query)

            if not nodes:
                logger.warning("No documents retrieved from vector store")

                if LANGFUSE_ENABLED:
                    langfuse_context.update_current_observation(
                        output={"chunks_found": 0, "message": "No documents retrieved"}
                    )

                return self._empty_response()

            logger.debug(f"Retrieved {len(nodes)} nodes from vector store")

            formatted_results = self._process_retrieved_nodes(nodes)

            if not formatted_results:
                logger.warning(f"No documents passed score threshold ({self.score_threshold:.3f})")
                logger.info(f"Try lowering the score threshold in the configuration")

                if LANGFUSE_ENABLED:
                    langfuse_context.update_current_observation(
                        output={
                            "chunks_found": 0,
                            "message": "No documents passed score threshold",
                            "threshold": self.score_threshold
                        }
                    )

                return self._empty_response()

            formatted_results = formatted_results[:self.top_k]

            context_data = self._build_context(formatted_results)

            logger.info(
                f"Retrieved {len(formatted_results)} relevant chunks "
                f"(scores: {formatted_results[0]['score']:.3f} - {formatted_results[-1]['score']:.3f})"
            )

            logger.info(f"DEBUG: Returning {len(context_data.get('chunks', []))} chunks")
            for idx, chunk in enumerate(context_data.get('chunks', [])):
                logger.info(f"   Chunk {idx + 1}: file_name='{chunk.get('file_name')}', "
                            f"text_len={len(chunk.get('text', ''))}, score={chunk.get('score')}")

            if LANGFUSE_ENABLED:
                scores = [r['score'] for r in formatted_results]
                langfuse_context.update_current_observation(
                    output={
                        "chunks_found": len(formatted_results),
                        "context_length": len(context_data.get('context', '')),
                        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
                        "max_score": round(max(scores), 4) if scores else 0,
                        "min_score": round(min(scores), 4) if scores else 0,
                        "sources": context_data.get('sources', [])
                    },
                    metadata={
                        "chunks_returned": len(context_data.get('chunks', [])),
                        "total_retrieved": len(nodes)
                    }
                )

            return context_data

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )

            return self._empty_response()

    def _process_retrieved_nodes(self, nodes) -> List[Dict[str, Any]]:
        """
        Process retrieved nodes and filter by score threshold

        Args:
            nodes: Retrieved nodes from LlamaIndex retriever

        Returns:
            List of formatted and filtered results
        """
        formatted_results = []

        if not nodes:
            return formatted_results

        logger.debug(f"Processing {len(nodes)} retrieved nodes")

        for idx, node in enumerate(nodes):
            if hasattr(node, 'node') and node.node:
                text = node.node.text or ""
                score = getattr(node, 'score', 0.0)
                metadata = getattr(node.node, 'metadata', {}) or {}

                logger.debug(f"  Node {idx + 1}: score={score:.4f}, text_len={len(text)}")

                if score >= self.score_threshold:
                    file_name = self._extract_filename(metadata)

                    result_data = {
                        "text": text,
                        "score": float(score),
                        "metadata": metadata,
                        "rank": idx + 1,
                        "source": metadata.get("file_path", file_name),
                        "file_name": file_name,
                        "page": metadata.get("page", ""),
                        "chunk_id": metadata.get("chunk_id", ""),
                    }

                    formatted_results.append(result_data)
                    logger.debug(f"    Added: file='{file_name}', score={score:.4f}")
                else:
                    logger.debug(f"    Filtered out (score < {self.score_threshold:.3f})")

        formatted_results.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(f"Kept {len(formatted_results)}/{len(nodes)} results after threshold filtering")
        return formatted_results

    def _extract_filename(self, metadata: Dict[str, Any]) -> str:
        """
        Extract filename from metadata with multiple fallback strategies

        Args:
            metadata: Metadata dictionary from the node

        Returns:
            Filename string (never empty or None)
        """
        if not metadata:
            return "Unknown Document"

        file_name = metadata.get("file_name")
        if file_name and str(file_name).strip():
            return str(file_name).strip()

        file_path = metadata.get("file_path", "")
        if file_path:
            if "/" in file_path:
                file_name = file_path.split("/")[-1]
            elif "\\" in file_path:
                file_name = file_path.split("\\")[-1]
            else:
                file_name = file_path

            if file_name and str(file_name).strip():
                return str(file_name).strip()

        source = metadata.get("source", "")
        if source:
            if "/" in source:
                file_name = source.split("/")[-1]
            elif "\\" in source:
                file_name = source.split("\\")[-1]
            else:
                file_name = source

            if file_name and str(file_name).strip():
                return str(file_name).strip()

        return "Unknown Document"

    def _build_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build final context from retrieved results

        Args:
            results: Processed and filtered results

        Returns:
            Dictionary with context, sources, and chunks
        """
        context_chunks = []
        for i, result in enumerate(results):
            chunk_text = result.get("text", "").strip()
            if chunk_text:
                if i > 0:
                    context_chunks.append("")

                file_name = result.get("file_name", "Unknown")
                source_label = f"[Source: {file_name}]"
                context_chunks.append(f"{source_label}\n{chunk_text}")

        context_text = "\n".join(context_chunks)

        sources = []
        chunks_data = []

        if self.include_source:
            source_set = set()
            for result in results:
                file_name = result.get("file_name", "Unknown")
                if file_name and file_name not in source_set and file_name != "Unknown Document":
                    source_set.add(file_name)
                    sources.append(file_name)

            for i, result in enumerate(results):
                chunk_obj = {
                    "file_name": result.get("file_name", "Unknown Document"),
                    "text": result.get("text", ""),
                    "score": float(result.get("score", 0.0)),
                    "rank": result.get("rank", i + 1),
                    "source": result.get("source", result.get("file_name", "Unknown Document")),
                    "page": result.get("page", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "file_path": result.get("source", ""),
                    "metadata": result.get("metadata", {})
                }

                if not chunk_obj["text"] or not chunk_obj["file_name"]:
                    logger.warning(f"Chunk {i + 1} missing required fields, skipping")
                    continue

                chunks_data.append(chunk_obj)

                logger.debug(
                    f"Chunk {i + 1}: file='{chunk_obj['file_name']}', "
                    f"score={chunk_obj['score']:.3f}, text_len={len(chunk_obj['text'])}"
                )

        scores = [r["score"] for r in results if "score" in r]

        final_data = {
            "context": context_text,
            "sources": sources,
            "chunks": chunks_data,
            "stats": {
                "total_chunks": len(results),
                "max_score": round(max(scores), 4) if scores else 0.0,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0
            }
        }

        logger.info(f"Built context with {len(chunks_data)} chunks for response")

        if chunks_data:
            logger.info(f"Sample chunk keys: {list(chunks_data[0].keys())}")

        return final_data

    def _empty_response(self) -> Dict[str, Any]:
        """Return empty response structure"""
        return {
            "context": "",
            "sources": [],
            "chunks": [],
            "stats": {
                "total_chunks": 0,
                "max_score": 0.0,
                "min_score": 0.0,
                "average_score": 0.0
            }
        }

    @observe(name="retrieve_with_scores")
    async def retrieve_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve results with detailed scores for analysis

        Args:
            query: User query string

        Returns:
            List of results with detailed scoring information
        """
        try:
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    input={"query": query}
                )

            if not query or not query.strip():
                return []

            query = query.strip()

            index = vector_store_manager.get_vector_store_index(
                self.store_id,
                self.store_name,
                self.config.embed_model
            )

            if not index:
                return []

            retriever = index.as_retriever(similarity_top_k=self.top_k)
            nodes = retriever.retrieve(query)

            formatted_results = []
            for idx, node in enumerate(nodes):
                if hasattr(node, 'node') and node.node:
                    score = getattr(node, 'score', 0.0)
                    if score >= self.score_threshold:
                        metadata = getattr(node.node, 'metadata', {}) or {}
                        result_data = {
                            "text": node.node.text or "",
                            "score": round(float(score), 4),
                            "rank": idx + 1,
                            "metadata": metadata,
                            "file_name": self._extract_filename(metadata)
                        }
                        formatted_results.append(result_data)

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={"results_count": len(formatted_results)}
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error in retrieve_with_scores: {e}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )

            return []

    def update_retrieval_parameters(
            self,
            top_k: Optional[int] = None,
            score_threshold: Optional[float] = None,
            include_source: Optional[bool] = None
    ):
        """
        Update retrieval parameters on the fly

        Args:
            top_k: New top_k value
            score_threshold: New score threshold
            include_source: Whether to include sources
        """
        if top_k is not None:
            self.top_k = max(1, min(top_k, 20))
        if score_threshold is not None:
            self.score_threshold = self._normalize_threshold(score_threshold)
        if include_source is not None:
            self.include_source = include_source

        logger.info(f"Updated retrieval parameters:")
        logger.info(
            f"   top_k={self.top_k}, threshold={self.score_threshold:.3f}, include_source={self.include_source}")

    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store

        Returns:
            Dictionary with store information
        """
        try:
            store_info = vector_store_manager.get_store_info(self.store_id, self.store_name)
            if store_info:
                logger.info(f"Store info: {store_info.get('document_count', 0)} documents")
            return store_info or {}
        except Exception as e:
            logger.error(f"Error getting store info: {e}")
            return {}

    @observe(name="test_connection")
    async def test_connection(self) -> bool:
        """
        Test if the retriever can connect to the vector store

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Testing connection to store: {self.store_name}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    input={"store_name": self.store_name, "store_id": self.store_id}
                )

            index = vector_store_manager.get_vector_store_index(
                self.store_id,
                self.store_name,
                self.config.embed_model
            )

            success = index is not None

            if success:
                logger.info(f"Connection test passed for store: {self.store_name}")
            else:
                logger.error(f"Connection test failed for store: {self.store_name}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={"success": success}
                )

            return success

        except Exception as e:
            logger.error(f"Connection test failed: {e}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e),
                    output={"success": False}
                )

            return False

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get current retrieval configuration and statistics

        Returns:
            Dictionary with retrieval stats
        """
        return {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "provider": self.config.provider,
            "model": self.config.model_name,
            "parameters": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "include_source": self.include_source
            }
        }
