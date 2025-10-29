import os
import json
import uuid
import logging
import warnings
import yaml
import requests
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GUARDRAILS_URL = "http://localhost:8001"
GUARDRAILS_ENABLED = True
GUARDRAILS_TIMEOUT = 10

try:
    from langfuse.decorators import observe, langfuse_context
    from langfuse import Langfuse

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    LANGFUSE_ENABLED = True
    logger.info("LangFuse initialized successfully")
except Exception as e:
    LANGFUSE_ENABLED = False
    langfuse_client = None
    logger.warning(f"LangFuse not initialized: {e}")


    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

DATA_DIR = "data"
VECTOR_STORES_FILE = os.path.join(DATA_DIR, "vector_stores.json")
EMBEDDING_CONFIG_FILE = os.path.join(DATA_DIR, "embedding_config.json")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

try:
    from ingestion.vector_store_manager import vector_store_manager
    from ingestion.indexing import Indexing
    from ingestion.config import Config
    from retrieval.retriever import Retriever
    from generation.generator import Generator

    INGESTION_AVAILABLE = True
    RAG_AVAILABLE = True
    logger.info("All modules loaded successfully")
except ImportError as e:
    INGESTION_AVAILABLE = False
    RAG_AVAILABLE = False
    logger.error(f"Failed to import modules: {e}")

generator = None
if RAG_AVAILABLE:
    try:
        generator = Generator(base_url="http://localhost:4000/v1", api_key="dummy-key")
        logger.info("Generator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        RAG_AVAILABLE = False
        generator = None


def validate_user_input(text: str, guard_name: str = "sensitive-topic-guard") -> tuple[bool, str]:
    """
    Validate user input against guardrails before processing

    Args:
        text: The user input text to validate
        guard_name: Name of the guard to use (default: sensitive-topic-guard)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passed, False if blocked
        - error_message: Empty string if valid, error description if blocked
    """
    if not GUARDRAILS_ENABLED:
        logger.warning("Guardrails disabled, allowing request")
        return True, ""

    try:
        logger.info(f"Validating user input with {guard_name}: {text[:50]}...")

        response = requests.post(
            f"{GUARDRAILS_URL}/guards/{guard_name}/validate",
            json={"text": text},
            timeout=GUARDRAILS_TIMEOUT
        )

        data = response.json()

        if response.status_code == 200 and data.get("validation_passed", False):
            logger.info(f"Guardrails validation PASSED")
            return True, ""
        else:
            error_msg = data.get("error", "Content violates safety guidelines")
            logger.warning(f"Guardrails validation FAILED: {error_msg[:100]}")
            return False, error_msg

    except requests.Timeout:
        logger.error(f"Guardrails timeout after {GUARDRAILS_TIMEOUT}s")
        return True, ""

    except requests.ConnectionError:
        logger.error(f"Guardrails connection error - service may be down")
        return True, ""

    except Exception as e:
        logger.error(f"Guardrails validation error: {e}")
        return True, ""


def init_vector_stores():
    """Initialize vector stores JSON file"""
    if not os.path.exists(VECTOR_STORES_FILE):
        with open(VECTOR_STORES_FILE, 'w') as f:
            json.dump({"vector_stores": []}, f, indent=2)
        logger.info("Created vector_stores.json")


def init_embedding_config():
    """Initialize embedding config JSON file"""
    if not os.path.exists(EMBEDDING_CONFIG_FILE):
        default_config = {
            "huggingface": {
                "embedding_models": [
                    "BAAI/bge-small-en",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/all-MiniLM-L6-v2"
                ]
            },
            "ollama": {
                "embedding_models": [
                    "nomic-embed-text",
                    "all-minilm"
                ]
            }
        }
        with open(EMBEDDING_CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info("Created embedding_config.json")


init_vector_stores()
init_embedding_config()


def load_vector_stores():
    """Load vector stores from JSON"""
    try:
        with open(VECTOR_STORES_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"vector_stores": []}


def save_vector_stores(data):
    """Save vector stores to JSON"""
    with open(VECTOR_STORES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_embedding_config():
    """Load embedding config from JSON"""
    try:
        with open(EMBEDDING_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


def find_store(store_id: str):
    """Find a vector store by ID"""
    data = load_vector_stores()
    for store in data["vector_stores"]:
        if store["id"] == store_id:
            return store
    return None


def load_litellm_config():
    """Load LiteLLM models from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        models = []
        for model in config.get('model_list', []):
            model_info = model.get('model_info', {})
            models.append({
                "model_name": model.get('model_name'),
                "litellm_provider": model_info.get('provider'),
                "mode": "chat",
                "description": model_info.get('description', ''),
                "context_window": model_info.get('context_window', 4096)
            })

        return {
            "models": models,
            "default_model": config.get('default_model', 'llama-3.3-70b-groq')
        }
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return {
            "models": [
                {
                    "model_name": "llama-3.3-70b-groq",
                    "litellm_provider": "groq",
                    "mode": "chat",
                    "description": "Fast 70B model",
                    "context_window": 32768
                }
            ],
            "default_model": "llama-3.3-70b-groq"
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AI RAG Dashboard Starting...")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Ingestion available: {INGESTION_AVAILABLE}")
    logger.info(f"RAG available: {RAG_AVAILABLE}")
    logger.info(f"LangFuse enabled: {LANGFUSE_ENABLED}")
    logger.info(f"Guardrails enabled: {GUARDRAILS_ENABLED}")

    if INGESTION_AVAILABLE:
        stores = load_vector_stores()
        logger.info(f"Loaded {len(stores['vector_stores'])} existing vector stores")

    logger.info("=" * 60)
    yield
    logger.info("Shutting down...")

    if LANGFUSE_ENABLED and langfuse_client:
        langfuse_client.flush()
        logger.info("LangFuse flushed")


app = FastAPI(title="AI RAG Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/data-ingestion", response_class=HTMLResponse)
async def data_ingestion(request: Request):
    return templates.TemplateResponse("ingestion.html", {"request": request})


@app.get("/chatbot-ui", response_class=HTMLResponse)
async def chatbot_ui(request: Request):
    store_id = request.query_params.get('store_id')
    usecase = request.query_params.get('usecase', 'Chat')

    if not store_id:
        raise HTTPException(status_code=400, detail="store_id is required")

    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "store_id": store_id,
        "usecase": usecase
    })


@app.get("/api/embedding-config")
async def get_embedding_config():
    """Get embedding providers and models"""
    config = load_embedding_config()
    return JSONResponse(content=config)


@app.get("/api/vectorstores")
async def get_vector_stores():
    """Get all vector stores with updated counts"""
    data = load_vector_stores()

    if INGESTION_AVAILABLE:
        for store in data["vector_stores"]:
            try:
                chunk_count = vector_store_manager.get_collection_count(
                    store["id"],
                    store["name"]
                )
                store["documentCount"] = chunk_count
            except Exception as e:
                logger.warning(f"Could not get count for {store['name']}: {e}")

        save_vector_stores(data)

    return JSONResponse(content=data)


@app.get("/api/vectorstores/{store_id}/verify")
async def verify_vector_store(store_id: str):
    """Verify if vector store exists"""
    store = find_store(store_id)

    if not store:
        return JSONResponse(
            status_code=404,
            content={
                "exists": False,
                "error": "Vector store not found"
            }
        )

    document_count = store.get("documentCount", 0)
    file_count = store.get("fileCount", len(store.get("files", [])))

    if INGESTION_AVAILABLE:
        try:
            document_count = vector_store_manager.get_collection_count(
                store_id,
                store["name"]
            )
        except Exception as e:
            logger.warning(f"Could not verify count: {e}")

    return JSONResponse(content={
        "exists": True,
        "has_embeddings": store.get("has_embeddings", False),
        "documentCount": document_count,
        "fileCount": file_count,
        "name": store.get("name", "Unknown"),
        "provider": store.get("provider", "Unknown"),
        "model": store.get("model", "Unknown")
    })


@app.get("/api/vectorstores/{store_id}/info")
async def get_vector_store_info(store_id: str):
    """Get detailed info about a vector store"""
    store = find_store(store_id)

    if not store:
        raise HTTPException(status_code=404, detail="Vector store not found")

    document_count = store.get("documentCount", 0)
    file_count = store.get("fileCount", len(store.get("files", [])))

    if INGESTION_AVAILABLE:
        try:
            document_count = vector_store_manager.get_collection_count(
                store_id,
                store["name"]
            )
        except Exception as e:
            logger.warning(f"Could not get info count: {e}")

    return JSONResponse(content={
        "id": store["id"],
        "name": store["name"],
        "provider": store["provider"],
        "model": store["model"],
        "description": store.get("description", ""),
        "files": store.get("files", []),
        "fileCount": file_count,
        "documentCount": document_count,
        "createdAt": store.get("createdAt", ""),
        "has_embeddings": store.get("has_embeddings", False)
    })


@app.post("/api/vectorstores")
async def create_vector_store(
        name: str = Form(...),
        provider: str = Form(...),
        model: str = Form(...),
        description: str = Form(""),
        files: List[UploadFile] = File([])
):
    """Create a new vector store with ChromaDB"""
    store_id = None

    try:
        data = load_vector_stores()

        for store in data["vector_stores"]:
            if store["name"].lower() == name.lower():
                raise HTTPException(status_code=400, detail="Vector store with this name already exists")

        store_id = str(uuid.uuid4())

        if not INGESTION_AVAILABLE:
            raise HTTPException(status_code=500, detail="Ingestion modules not available")

        success = vector_store_manager.create_vector_store(store_id, name, provider, model)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create ChromaDB collection")

        logger.info(f"Created ChromaDB collection for: {name}")

        saved_files = []
        file_paths = []

        for file in files:
            filename = f"{store_id}_{file.filename}"
            file_path = os.path.join(UPLOADS_DIR, filename)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            saved_files.append(file.filename)
            file_paths.append(file_path)

        logger.info(f"Saved {len(saved_files)} files")

        chunk_count = 0
        has_embeddings = False

        if file_paths:
            try:
                config = Config(provider=provider, model_name=model)

                chunk_count = await Indexing.index_files(
                    store_id,
                    name,
                    file_paths,
                    config
                )

                has_embeddings = chunk_count > 0
                logger.info(f"Indexed {chunk_count} chunks into ChromaDB")

            except Exception as e:
                logger.error(f"Error during indexing: {e}")

        collection_name = vector_store_manager._get_collection_name(name)

        new_store = {
            "id": store_id,
            "name": name,
            "provider": provider,
            "model": model,
            "description": description,
            "files": saved_files,
            "fileCount": len(saved_files),
            "documentCount": chunk_count,
            "chroma_collection_name": collection_name,
            "has_embeddings": has_embeddings,
            "createdAt": datetime.now().isoformat()
        }

        data["vector_stores"].append(new_store)
        save_vector_stores(data)

        logger.info(f"Vector store created successfully: {name}")
        logger.info(f"  Files: {len(saved_files)}, Chunks: {chunk_count}")

        return JSONResponse(content=new_store, status_code=201)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")

        if store_id and INGESTION_AVAILABLE:
            try:
                vector_store_manager.delete_vector_store(store_id, name)
                logger.info("Cleaned up failed vector store creation")
            except:
                pass

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vectorstores/{store_id}/files")
async def upload_files_to_store(
        store_id: str,
        files: List[UploadFile] = File(...)
):
    """Upload additional files to existing vector store"""
    try:
        store = find_store(store_id)

        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if not INGESTION_AVAILABLE:
            raise HTTPException(status_code=500, detail="Ingestion modules not available")

        saved_files = []
        file_paths = []

        for file in files:
            filename = f"{store_id}_{file.filename}"
            file_path = os.path.join(UPLOADS_DIR, filename)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            saved_files.append(file.filename)
            file_paths.append(file_path)

        logger.info(f"Saved {len(saved_files)} new files")

        chunk_count = 0
        if file_paths:
            try:
                config = Config(provider=store["provider"], model_name=store["model"])

                chunk_count = await Indexing.index_files(
                    store_id,
                    store["name"],
                    file_paths,
                    config
                )

                logger.info(f"Indexed {chunk_count} new chunks")

            except Exception as e:
                logger.error(f"Error indexing new files: {e}")

        data = load_vector_stores()
        store_found = False

        for i, s in enumerate(data["vector_stores"]):
            if s["id"] == store_id:
                data["vector_stores"][i]["files"].extend(saved_files)
                data["vector_stores"][i]["fileCount"] = len(data["vector_stores"][i]["files"])

                try:
                    total_count = vector_store_manager.get_collection_count(
                        store_id,
                        store["name"]
                    )
                    data["vector_stores"][i]["documentCount"] = total_count
                except:
                    data["vector_stores"][i]["documentCount"] = data["vector_stores"][i].get("documentCount",
                                                                                             0) + chunk_count

                data["vector_stores"][i]["has_embeddings"] = True
                store_found = True

                save_vector_stores(data)

                logger.info(f"Updated vector store: {store['name']}")

                return JSONResponse(content={
                    "message": "Files uploaded successfully",
                    "files_uploaded": len(saved_files),
                    "chunks_added": chunk_count,
                    "total_files": data["vector_stores"][i]["fileCount"],
                    "total_documents": data["vector_stores"][i]["documentCount"]
                })

        if not store_found:
            raise HTTPException(status_code=404, detail="Store not found in database")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/vectorstores/{store_id}")
async def delete_vector_store(store_id: str):
    """Delete a vector store"""
    try:
        store = find_store(store_id)

        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if INGESTION_AVAILABLE:
            try:
                vector_store_manager.delete_vector_store(store_id, store["name"])
                logger.info(f"Deleted ChromaDB collection: {store['name']}")
            except Exception as e:
                logger.warning(f"ChromaDB deletion warning: {e}")

        for filename in store.get("files", []):
            file_path = os.path.join(UPLOADS_DIR, f"{store_id}_{filename}")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file: {filename}")
            except Exception as e:
                logger.warning(f"Could not delete file {filename}: {e}")

        data = load_vector_stores()
        data["vector_stores"] = [s for s in data["vector_stores"] if s["id"] != store_id]
        save_vector_stores(data)

        logger.info(f"Deleted vector store: {store['name']}")

        return JSONResponse(content={"message": "Vector store deleted successfully"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/litellm/models")
async def get_litellm_models():
    """Get available LLM models from config.yaml"""
    return JSONResponse(content=load_litellm_config())


@app.post("/api/chat")
@observe(name="rag_chat_endpoint", as_type="generation")
async def chat(request: Request):
    """Complete RAG Chat endpoint with INPUT VALIDATION + retrieval and generation + LangFuse tracing"""
    try:
        body = await request.json()
        message = body.get("message", "")
        store_id = body.get("store_id", "")
        config = body.get("config", {})

        if not message or not store_id:
            raise HTTPException(status_code=400, detail="message and store_id required")

        logger.info("Running guardrails validation on user input...")
        is_valid, error_msg = validate_user_input(message)

        if not is_valid:
            logger.warning(f"BLOCKED toxic input: {message[:50]}...")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_trace(
                    name="RAG Chat - BLOCKED",
                    tags=["chat", "blocked", "toxic"],
                    metadata={"blocked_reason": error_msg}
                )
                langfuse_context.update_current_observation(
                    input={"query": message[:100]},
                    output={"blocked": True, "reason": error_msg},
                    level="WARNING"
                )

            return JSONResponse(content={
                "response": "I cannot process this request as it violates our content safety policy. Please rephrase your message in a respectful manner.",
                "source_documents": [],
                "chunks": [],
                "stats": {
                    "blocked": True,
                    "reason": "Content safety violation"
                },
                "blocked": True,
                "error": error_msg
            }, status_code=200)

        logger.info("Guardrails validation passed - proceeding with RAG pipeline")

        if not RAG_AVAILABLE or generator is None:
            raise HTTPException(
                status_code=503,
                detail="RAG system not available. Please ensure LiteLLM proxy is running on port 4000"
            )

        store = find_store(store_id)
        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if not store.get("has_embeddings", False):
            raise HTTPException(status_code=400, detail="Vector store has no embeddings")

        model = config.get("model", "llama-3.3-70b-groq")
        temperature = float(config.get("temperature", 0.3))
        max_tokens = int(config.get("max_tokens", 1000))
        top_k = int(config.get("top_k", 5))
        score_threshold = float(config.get("score_threshold", 0.2))
        include_sources = config.get("include_sources", True)

        if LANGFUSE_ENABLED:
            langfuse_context.update_current_trace(
                name="RAG Chat",
                user_id=store_id,
                session_id=store_id,
                tags=["chat", "rag", model, store["name"]],
                metadata={
                    "store_name": store["name"],
                    "store_provider": store["provider"],
                    "store_model": store["model"],
                    "query_length": len(message),
                    "guardrails_passed": True,
                    "config": {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_k": top_k,
                        "score_threshold": score_threshold
                    }
                }
            )

            langfuse_context.update_current_observation(
                input={"query": message}
            )

        logger.info(f"Chat request for store: {store['name']}")
        logger.info(f"User query: {message[:100]}...")
        logger.info(f"Config: model={model}, temp={temperature}, max_tokens={max_tokens}")
        logger.info(f"Retrieval: top_k={top_k}, threshold={score_threshold}")

        retriever = Retriever(
            store_id=store_id,
            store_name=store["name"],
            provider=store["provider"],
            model_name=store["model"],
            top_k=top_k,
            score_threshold=score_threshold,
            include_source=include_sources
        )

        logger.info("Retrieving relevant context...")
        context_data = await retriever.retrieve_context(message)

        if not context_data["context"]:
            logger.warning("No relevant context found")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={"no_context": True},
                    level="WARNING"
                )

            return JSONResponse(content={
                "response": "I couldn't find any relevant information in the knowledge base to answer your question. Try lowering the score threshold or asking a different question.",
                "source_documents": [],
                "chunks": [],
                "stats": context_data["stats"],
                "model": model,
                "has_context": False
            })

        logger.info(f"Retrieved {context_data['stats']['total_chunks']} relevant chunks")

        logger.info(f"Generating response with model: {model}")
        generation_result = generator.generate_response(
            context=context_data["context"],
            query=message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        chunks_to_return = context_data["chunks"] if include_sources else []

        validated_chunks = []
        for idx, chunk in enumerate(chunks_to_return):
            file_name = str(chunk.get("file_name", "")).strip()
            text = str(chunk.get("text", "")).strip()
            score = chunk.get("score")

            if not file_name or not text or score is None:
                logger.warning(f"Skipping chunk {idx + 1}: missing required fields")
                continue

            validated_chunk = {
                "file_name": file_name if file_name else "Unknown Document",
                "text": text,
                "score": float(score),
                "source": chunk.get("source", file_name),
                "page": chunk.get("page", ""),
                "rank": chunk.get("rank", idx + 1),
                "chunk_id": chunk.get("chunk_id", ""),
            }
            validated_chunks.append(validated_chunk)

        response_data = {
            "response": generation_result["response"],
            "source_documents": context_data["sources"] if include_sources else [],
            "chunks": validated_chunks,
            "stats": {
                **context_data["stats"],
                "model": generation_result["model_used"],
                "include_sources": include_sources,
                "chunks_returned": len(validated_chunks),
                "guardrails_passed": True
            },
            "has_context": True,
            "blocked": False
        }

        if "stats" in generation_result:
            response_data["stats"].update(generation_result["stats"])

        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                output={
                    "response": generation_result["response"],
                    "response_length": len(generation_result["response"]),
                    "chunks_used": len(validated_chunks),
                    "sources": context_data["sources"]
                },
                metadata={
                    "retrieval_stats": context_data["stats"],
                    "generation_stats": generation_result.get("stats", {}),
                    "hallucination_detected": generation_result.get("hallucination_detected", False)
                },
                level="DEFAULT"
            )

        logger.info(f"Chat completed successfully")
        logger.info(f"Returning {len(validated_chunks)} chunks to frontend")

        return JSONResponse(content=response_data)

    except HTTPException as http_exc:
        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"HTTP {http_exc.status_code}: {http_exc.detail}"
            )
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)

        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e)
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-langfuse")
@observe(name="test_langfuse_connection")
async def test_langfuse():
    """Test LangFuse connection and tracing"""
    try:
        if not LANGFUSE_ENABLED or not langfuse_client:
            return JSONResponse(content={
                "status": "disabled",
                "message": "LangFuse is not enabled. Check your .env file."
            })

        langfuse_context.update_current_trace(
            name="Test Trace",
            tags=["test", "health-check"]
        )

        langfuse_context.update_current_observation(
            input={"test": "connection"},
            output={"status": "success"}
        )

        langfuse_client.flush()

        return JSONResponse(content={
            "status": "success",
            "message": "LangFuse is working! Check your LangFuse dashboard for this trace.",
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": f"LangFuse test failed: {str(e)}"
        }, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "ingestion_available": INGESTION_AVAILABLE,
        "rag_available": RAG_AVAILABLE,
        "langfuse_enabled": LANGFUSE_ENABLED,
        "guardrails_enabled": GUARDRAILS_ENABLED,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
