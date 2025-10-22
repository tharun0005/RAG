import os
import uuid
import re
import unicodedata
from typing import List, Dict
import logging
from .config import Config
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class Indexing:
    @staticmethod
    def clean_text_for_embedding(text: str) -> str:
        """Clean and sanitize text for embedding generation"""
        if not text or not isinstance(text, str):
            return ""

        try:
            # Remove null bytes
            text = text.replace('\x00', '')

            # Remove control characters except newline, tab, and carriage return
            text = ''.join(
                char for char in text
                if unicodedata.category(char)[0] != 'C' or char in '\n\t\r '
            )

            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)

            # Remove non-printable characters except whitespace
            text = ''.join(c for c in text if c.isprintable() or c.isspace())

            # Replace multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text)

            # Remove excessive newlines (more than 2 consecutive)
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Strip leading/trailing whitespace
            text = text.strip()

            # Ensure minimum length (avoid embedding empty/tiny chunks)
            if len(text) < 10:
                return ""

            # Truncate if too long (some models have limits)
            if len(text) > 8000:
                text = text[:8000]
                logger.debug("Truncated text to 8000 characters")

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""

    @staticmethod
    def validate_node(node: TextNode) -> bool:
        """Validate that a node is suitable for embedding"""
        if not node or not hasattr(node, 'text'):
            logger.debug("Node has no text attribute")
            return False

        text = node.text

        if not text or not isinstance(text, str):
            logger.debug("Node text is None or not a string")
            return False

        if len(text.strip()) < 10:
            logger.debug(f"Node text too short: {len(text)} chars")
            return False

        # Check if text has some alphanumeric content
        if not any(c.isalnum() for c in text):
            logger.debug("Node text has no alphanumeric characters")
            return False

        # Check for null bytes
        if '\x00' in text:
            logger.debug("Node text contains null bytes")
            return False

        return True

    @staticmethod
    async def create_vector_store(store_data: Dict, files: List[str] = None):
        """Create a new vector store and optionally index files using LlamaIndex"""
        try:
            from .vector_store_manager import vector_store_manager

            logger.info(f"Creating vector store: {store_data['name']}")

            # Create config for embedding model
            config = Config(
                provider=store_data["provider"],
                model_name=store_data["model"]
            )

            # Create vector store in Chroma
            success = vector_store_manager.create_vector_store(
                store_id=store_data["id"],
                store_name=store_data["name"],
                provider=store_data["provider"],
                model=store_data["model"]
            )

            if not success:
                logger.error("Failed to create vector store in Chroma")
                return 0

            logger.info(f"✓ Vector store created in ChromaDB: {store_data['name']}")

            # If files are provided, index them
            chunk_count = 0
            if files:
                logger.info(f"Indexing {len(files)} files into vector store")
                chunk_count = await Indexing.index_files(
                    store_data["id"],
                    store_data["name"],
                    files,
                    config
                )
                logger.info(f"✓ Indexing completed. Created {chunk_count} chunks.")
            else:
                logger.info("No files provided for indexing")

            return chunk_count

        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return 0

    @staticmethod
    async def update_vector_store(store_data: Dict, new_files: List[str]):
        """Update existing vector store with new files"""
        try:
            logger.info(f"Updating vector store {store_data['name']} with {len(new_files)} new files")

            config = Config(
                provider=store_data["provider"],
                model_name=store_data["model"]
            )

            chunk_count = await Indexing.index_files(
                store_data["id"],
                store_data["name"],
                new_files,
                config
            )
            logger.info(f"✓ Update completed. Added {chunk_count} new chunks.")
            return chunk_count

        except Exception as e:
            logger.error(f"Error updating vector store: {e}", exc_info=True)
            return 0

    @staticmethod
    async def delete_vector_store(store_id: str, store_name: str):
        """Delete vector store from ChromaDB"""
        try:
            from .vector_store_manager import vector_store_manager

            logger.info(f"Deleting vector store: {store_name} (ID: {store_id})")
            return vector_store_manager.delete_vector_store(store_id, store_name)
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}", exc_info=True)
            return False

    @staticmethod
    async def index_files(store_id: str, store_name: str, file_paths: List[str], config: Config):
        """Index files into vector store using LlamaIndex with robust error handling"""
        try:
            from .vector_store_manager import vector_store_manager

            all_nodes = []
            total_raw_chunks = 0
            skipped_chunks = 0

            logger.info(f"Processing {len(file_paths)} files for store: {store_name}")

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                try:
                    file_name = os.path.basename(file_path)
                    logger.info(f"📄 Processing file: {file_name}")

                    # Load document using SimpleDirectoryReader
                    loader = SimpleDirectoryReader(input_files=[file_path])
                    documents = loader.load_data()

                    if not documents:
                        logger.warning(f"No documents loaded from: {file_name}")
                        continue

                    logger.info(f"   Loaded {len(documents)} document(s)")

                    # Split documents into chunks
                    for doc_idx, doc in enumerate(documents):
                        try:
                            nodes = config.text_splitter.get_nodes_from_documents([doc])
                            total_raw_chunks += len(nodes)

                            logger.info(f"   Document {doc_idx + 1}: {len(nodes)} raw chunks")

                            # Process and validate each node
                            for node_idx, node in enumerate(nodes):
                                try:
                                    # Clean the text
                                    cleaned_text = Indexing.clean_text_for_embedding(node.text)

                                    if not cleaned_text:
                                        skipped_chunks += 1
                                        logger.debug(f"   Skipped chunk {node_idx + 1}: empty after cleaning")
                                        continue

                                    # Create TextNode with cleaned text and metadata
                                    text_node = TextNode(
                                        text=cleaned_text,
                                        metadata={
                                            **node.metadata,
                                            "file_path": file_path,
                                            "file_name": file_name,
                                            "chunk_id": str(uuid.uuid4()),
                                            "store_id": store_id,
                                            "store_name": store_name,
                                            "chunk_index": node_idx,
                                            "original_length": len(node.text),
                                            "cleaned_length": len(cleaned_text)
                                        }
                                    )

                                    # Final validation
                                    if Indexing.validate_node(text_node):
                                        all_nodes.append(text_node)
                                    else:
                                        skipped_chunks += 1
                                        logger.debug(f"   Skipped chunk {node_idx + 1}: failed validation")

                                except Exception as e:
                                    skipped_chunks += 1
                                    logger.warning(f"   Error processing chunk {node_idx + 1}: {e}")
                                    continue

                            logger.info(f"   ✓ Kept {len(all_nodes)} valid chunks from document {doc_idx + 1}")

                        except Exception as e:
                            logger.error(f"   Error splitting document {doc_idx + 1}: {e}")
                            continue

                    logger.info(f"✓ Processed: {file_name} -> {len(all_nodes)} valid chunks (skipped {skipped_chunks})")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                    continue

            # Log summary
            logger.info(f"📊 Processing Summary:")
            logger.info(f"   Total raw chunks: {total_raw_chunks}")
            logger.info(f"   Valid chunks: {len(all_nodes)}")
            logger.info(f"   Skipped chunks: {skipped_chunks}")

            # Add nodes to ChromaDB in batches
            if all_nodes:
                logger.info(f"🔄 Adding {len(all_nodes)} chunks to ChromaDB in batches...")

                success = vector_store_manager.add_nodes(
                    store_id,
                    store_name,
                    all_nodes,
                    config.embed_model
                )

                if success:
                    logger.info(f"✅ Successfully indexed {len(all_nodes)} chunks into {store_name}")
                    return len(all_nodes)
                else:
                    logger.error(f"❌ Failed to add chunks to {store_name}")
                    return 0
            else:
                logger.warning("⚠️ No valid chunks were created from files")
                logger.info(f"   Total raw chunks extracted: {total_raw_chunks}")
                logger.info(f"   All chunks were filtered out during cleaning/validation")
                return 0

        except Exception as e:
            logger.error(f"❌ Critical error indexing files: {e}", exc_info=True)
            return 0

    @staticmethod
    def get_chunk_count(store_id: str, store_name: str) -> int:
        """Get current chunk count from ChromaDB"""
        try:
            from .vector_store_manager import vector_store_manager
            return vector_store_manager.get_collection_count(store_id, store_name)
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
