import os
import logging
import shutil
import time
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode, TextNode
import chromadb
from chromadb.config import Settings
import re

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, base_path: str = "data/chroma_db"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.client_cache = {}
        self.vector_store_cache = {}

        logger.info(f"VectorStoreManager initialized at: {base_path}")

    def _get_store_path(self, store_id: str, store_name: str) -> str:
        """Get dedicated path for this store"""
        clean_name = self._get_collection_name(store_name)
        return os.path.join(self.base_path, f"{store_id}_{clean_name}")

    def _get_client(self, store_id: str, store_name: str):
        """Get or create ChromaDB client for this store"""
        if store_id in self.client_cache:
            return self.client_cache[store_id]

        store_path = self._get_store_path(store_id, store_name)
        os.makedirs(store_path, exist_ok=True)

        client = chromadb.PersistentClient(
            path=store_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.client_cache[store_id] = client
        return client

    def _close_client(self, store_id: str):
        """Close and remove client from cache"""
        try:
            if store_id in self.client_cache:
                del self.client_cache[store_id]
                logger.debug(f"Closed client for store_id: {store_id}")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")

    def create_vector_store(self, store_id: str, store_name: str, provider: str, model: str):
        """Create a new ChromaVectorStore with dedicated folder"""
        try:
            client = self._get_client(store_id, store_name)
            collection_name = self._get_collection_name(store_name)

            chroma_collection = client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "store_id": store_id,
                    "store_name": store_name,
                    "provider": provider,
                    "model": model,
                    "created_at": self._get_current_timestamp()
                }
            )

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            self.vector_store_cache[store_id] = vector_store

            store_path = self._get_store_path(store_id, store_name)
            logger.info(f"Created ChromaVectorStore: {collection_name} at {store_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False

    def get_vector_store(self, store_id: str, store_name: str) -> Optional[ChromaVectorStore]:
        """Get an existing ChromaVectorStore"""
        try:
            if store_id in self.vector_store_cache:
                return self.vector_store_cache[store_id]

            client = self._get_client(store_id, store_name)
            collection_name = self._get_collection_name(store_name)

            chroma_collection = client.get_collection(name=collection_name)

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            self.vector_store_cache[store_id] = vector_store

            logger.info(f"Retrieved ChromaVectorStore: {collection_name}")
            return vector_store

        except Exception as e:
            logger.error(f"Error getting vector store {store_name}: {e}")
            return None

    def add_nodes(self, store_id: str, store_name: str, nodes: List[BaseNode], embed_model):
        """Add nodes to vector store using LlamaIndex with batch processing"""
        try:
            vector_store = self.get_vector_store(store_id, store_name)
            if not vector_store:
                logger.error(f"Vector store not found: {store_name}")
                return False

            batch_size = 50
            total_nodes = len(nodes)
            successful_batches = 0

            logger.info(f"Adding {total_nodes} nodes in batches of {batch_size}")

            for i in range(0, total_nodes, batch_size):
                batch = nodes[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_nodes + batch_size - 1) // batch_size

                try:
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    VectorStoreIndex(
                        batch,
                        storage_context=storage_context,
                        embed_model=embed_model,
                        show_progress=True
                    )

                    successful_batches += 1
                    logger.info(f"Batch {batch_num}/{total_batches}: Added {len(batch)} nodes")

                except Exception as e:
                    logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
                    continue

            if successful_batches > 0:
                logger.info(f"Successfully added {successful_batches * batch_size} nodes to {store_name}")
                return True
            else:
                logger.error(f"All batches failed for {store_name}")
                return False

        except Exception as e:
            logger.error(f"Error adding nodes: {e}", exc_info=True)
            return False

    def get_vector_store_index(self, store_id: str, store_name: str, embed_model) -> Optional[VectorStoreIndex]:
        """Get VectorStoreIndex for querying"""
        try:
            vector_store = self.get_vector_store(store_id, store_name)
            if not vector_store:
                return None

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )

            logger.info(f"Created index for {store_name}")
            return index

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return None

    def query_vector_store(self, store_id: str, store_name: str, query: str, embed_model, top_k: int = 5):
        """Query the vector store"""
        try:
            index = self.get_vector_store_index(store_id, store_name, embed_model)
            if not index:
                return None

            query_engine = index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)

            logger.info(f"Queried {store_name} with top_k={top_k}")
            return response

        except Exception as e:
            logger.error(f"Error querying: {e}")
            return None

    def get_collection_count(self, store_id: str, store_name: str) -> int:
        """Get number of chunks in the collection"""
        try:
            client = self._get_client(store_id, store_name)
            collection_name = self._get_collection_name(store_name)
            collection = client.get_collection(name=collection_name)
            count = collection.count()

            logger.info(f"Store '{store_name}' has {count} chunks")
            return count

        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def delete_vector_store(self, store_id: str, store_name: str):
        """Delete vector store and its folder completely"""
        try:
            collection_name = self._get_collection_name(store_name)
            store_path = self._get_store_path(store_id, store_name)

            logger.info(f"Deleting vector store: {store_name} (ID: {store_id})")
            logger.info(f"   Collection: {collection_name}")
            logger.info(f"   Path: {store_path}")

            if store_id in self.vector_store_cache:
                del self.vector_store_cache[store_id]
                logger.debug("   Removed from vector store cache")

            try:
                client = self._get_client(store_id, store_name)
                client.delete_collection(name=collection_name)
                logger.info(f"   Deleted ChromaDB collection: {collection_name}")
            except Exception as e:
                logger.warning(f"   Could not delete collection (may not exist): {e}")

            self._close_client(store_id)
            logger.debug("   Closed ChromaDB client")

            import gc
            gc.collect()
            time.sleep(0.5)

            if os.path.exists(store_path):
                try:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            shutil.rmtree(store_path, ignore_errors=False)
                            logger.info(f"   Deleted folder: {store_path}")
                            break
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"   Retry {attempt + 1}/{max_retries}: {e}")
                                time.sleep(1)
                                gc.collect()
                            else:
                                shutil.rmtree(store_path, ignore_errors=True)
                                logger.warning(f"   Forced deletion with ignore_errors")
                except Exception as e:
                    logger.error(f"   Could not delete folder: {e}")
                    try:
                        self._force_delete_folder(store_path)
                        logger.info(f"   Forcefully deleted folder using alternative method")
                    except Exception as e2:
                        logger.error(f"   Alternative deletion also failed: {e2}")
            else:
                logger.warning(f"   Folder does not exist: {store_path}")

            logger.info(f"Successfully deleted vector store: {store_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting vector store: {e}", exc_info=True)
            return False

    def _force_delete_folder(self, folder_path: str):
        """Force delete folder using OS-specific commands (last resort)"""
        import platform
        import subprocess

        if platform.system() == "Windows":
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", folder_path],
                           check=False, capture_output=True)
        else:
            subprocess.run(["rm", "-rf", folder_path],
                           check=False, capture_output=True)

    def list_vector_stores(self):
        """List all vector stores"""
        try:
            stores = []

            if not os.path.exists(self.base_path):
                return stores

            for folder_name in os.listdir(self.base_path):
                folder_path = os.path.join(self.base_path, folder_name)

                if os.path.isdir(folder_path) and '_' in folder_name:
                    try:
                        store_id = folder_name.split('_')[0]
                        client = chromadb.PersistentClient(path=folder_path)
                        collections = client.list_collections()

                        for collection in collections:
                            metadata = collection.metadata or {}
                            stores.append({
                                "store_id": store_id,
                                "store_name": metadata.get("store_name", "unknown"),
                                "provider": metadata.get("provider", "unknown"),
                                "model": metadata.get("model", "unknown"),
                                "document_count": collection.count(),
                                "created_at": metadata.get("created_at", "unknown")
                            })
                    except:
                        pass

            logger.info(f"Found {len(stores)} vector stores")
            return stores

        except Exception as e:
            logger.error(f"Error listing stores: {e}")
            return []

    def get_store_info(self, store_id: str, store_name: str):
        """Get store information"""
        try:
            client = self._get_client(store_id, store_name)
            collection_name = self._get_collection_name(store_name)
            collection = client.get_collection(name=collection_name)

            return {
                "store_id": store_id,
                "store_name": store_name,
                "document_count": collection.count(),
                "metadata": collection.metadata
            }

        except Exception as e:
            logger.error(f"Error getting store info: {e}")
            return None

    def _get_collection_name(self, store_name: str) -> str:
        """Convert store name to valid collection name"""
        clean_name = store_name.lower().replace(' ', '_').replace('-', '_')
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', clean_name)
        clean_name = re.sub(r'^[^a-zA-Z0-9]+', '', clean_name)
        clean_name = re.sub(r'[^a-zA-Z0-9]+$', '', clean_name)

        if len(clean_name) < 3:
            clean_name = "vs_" + clean_name

        if len(clean_name) > 63:
            clean_name = clean_name[:63]

        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$', clean_name):
            import hashlib
            store_hash = hashlib.md5(store_name.encode()).hexdigest()[:8]
            clean_name = f"vs_{store_hash}"

        return clean_name

    def _get_current_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


vector_store_manager = VectorStoreManager()
