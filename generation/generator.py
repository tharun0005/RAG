import logging
import os
import warnings
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*getaddrinfo failed.*")

# Initialize LangFuse with error suppression
try:
    from langfuse import Langfuse

    # ✅ Configure with longer timeout and error suppression
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        timeout=60,  # ✅ Increase timeout to 60 seconds
        max_retries=3,  # ✅ Retry failed uploads
        flush_at=1,  # ✅ Flush after each trace
        flush_interval=5  # ✅ Flush every 5 seconds
    )
    LANGFUSE_ENABLED = True

    # ✅ Suppress LangFuse's internal logging
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    logger.info("✅ LangFuse initialized for Generator")
except Exception as e:
    LANGFUSE_ENABLED = False
    logger.warning(f"⚠️ LangFuse not initialized: {e}")


class Generator:
    """Enhanced generator for RAG QA using LiteLLM proxy"""

    def __init__(self, base_url: str = "http://localhost:4000/v1", api_key: str = "dummy-key"):
        """
        Initialize generator with LiteLLM proxy

        Args:
            base_url: LiteLLM proxy URL (default: http://localhost:4000/v1)
            api_key: API key (can be dummy for local proxy)
        """
        # ✅ Use OpenAI client pointing to LiteLLM proxy
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        logger.info(f"✅ Generator initialized with LiteLLM proxy at: {base_url}")

    @observe(name="generate_response", capture_input=True, capture_output=True)
    def generate_response(
            self,
            context: str,
            query: str,
            model: str = "llama-3.3-70b-groq",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate QA response with strict context adherence"""
        try:
            # Update trace metadata
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_trace(
                    tags=["generation", "rag", model],
                    metadata={
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )

            # Validate inputs
            if not context or not context.strip():
                logger.warning("⚠️ Empty context provided")
                return {
                    "response": "I don't have any relevant information in my knowledge base to answer that question.",
                    "model_used": model,
                    "has_context": False
                }

            if not query or not query.strip():
                logger.warning("⚠️ Empty query provided")
                return {
                    "response": "Please provide a question.",
                    "model_used": model,
                    "has_context": bool(context)
                }

            # Build prompts
            if not system_prompt:
                system_prompt = self._build_system_prompt()

            user_prompt = self._build_user_prompt(context, query)

            logger.info(f"🤖 Generating response using model: {model}")

            # Create LangFuse generation span
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    input={
                        "query": query,
                        "context_length": len(context),
                        "system_prompt_length": len(system_prompt)
                    }
                )

            # ✅ Call through OpenAI client (which points to LiteLLM proxy)
            # The proxy will handle routing to the correct provider
            response = self.client.chat.completions.create(
                model=model,  # Use model name from config.yaml
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            response_content = response.choices[0].message.content.strip()

            # Validate response
            is_hallucination = self._is_hallucination(response_content)
            if is_hallucination:
                logger.warning("⚠️ Potential hallucination detected")
                response_content = "I don't have enough information in the provided context to answer that question accurately."

            # Extract token usage
            usage = response.usage if hasattr(response, 'usage') else None
            tokens_used = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            result = {
                "response": response_content,
                "model_used": model,
                "has_context": True,
                "hallucination_detected": is_hallucination,
                "stats": {
                    "prompt_length": len(system_prompt) + len(user_prompt),
                    "response_length": len(response_content),
                    "context_length": len(context),
                    "temperature": temperature,
                    "tokens_used": tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }

            # Update LangFuse with output
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={
                        "response": response_content,
                        "response_length": len(response_content),
                        "hallucination_detected": is_hallucination
                    },
                    metadata={
                        "tokens_used": tokens_used,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens
                    }
                )

            logger.info(f"✅ Generated response ({len(response_content)} chars, {tokens_used} tokens)")
            return result

        except Exception as e:
            logger.error(f"❌ Error generating response: {e}", exc_info=True)

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )

            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "model_used": model,
                "error": str(e),
                "has_context": bool(context)
            }

    def _build_system_prompt(self) -> str:
        """Build strict system prompt to prevent hallucinations"""
        return """You are a precise AI assistant that answers questions STRICTLY based on provided context.

**CRITICAL RULES - YOU MUST FOLLOW THESE:**

1. **ONLY use information from the CONTEXT section below**
2. **NEVER use your general knowledge or training data**
3. **If the context doesn't contain the answer, you MUST say: "I don't have enough information in the provided context to answer that question."**
4. **DO NOT make assumptions or inferences beyond what's explicitly stated**
5. **DO NOT fabricate, guess, or hallucinate information**
6. **If you're unsure, say you don't know**

**Response Guidelines:**
- Be concise and direct
- Quote or reference specific parts of the context when possible
- If the context is partially relevant, acknowledge what you CAN answer and what you CANNOT
- Use phrases like "According to the context..." or "Based on the provided information..."

**Examples of CORRECT responses:**
✓ "I don't have enough information in the provided context to answer that question."
✓ "According to the context, [specific answer from context]."
✓ "The context mentions [X] but doesn't provide information about [Y]."

**Examples of INCORRECT responses:**
✗ Answering from general knowledge when context doesn't have the answer
✗ Making assumptions beyond what's explicitly stated
✗ Providing information not found in the context"""

    def _build_user_prompt(self, context: str, query: str) -> str:
        """Build user prompt with context and query"""
        return f"""**CONTEXT:**
{context}

**QUESTION:**
{query}

**INSTRUCTIONS:**
Answer the question using ONLY the information from the CONTEXT above. If the context doesn't contain relevant information to answer the question, respond with: "I don't have enough information in the provided context to answer that question."

**ANSWER:**"""

    def _is_hallucination(self, response: str) -> bool:
        """Simple heuristic to detect potential hallucinations"""
        response_lower = response.lower()

        hallucination_indicators = [
            "i think",
            "probably",
            "it might be",
            "i believe",
            "in general",
            "typically",
            "usually",
            "as far as i know",
            "from my knowledge",
            "based on my training"
        ]

        return any(indicator in response_lower for indicator in hallucination_indicators)

    @observe(name="stream_response")
    def stream_response(
            self,
            context: str,
            query: str,
            model: str = "llama-3.3-70b-groq",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            system_prompt: Optional[str] = None
    ):
        """Stream QA response from LLM"""
        try:
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_trace(
                    tags=["streaming", "rag", model]
                )

            if not context or not context.strip():
                yield "I don't have any relevant information in my knowledge base to answer that question."
                return

            if not query or not query.strip():
                yield "Please provide a question."
                return

            if not system_prompt:
                system_prompt = self._build_system_prompt()

            user_prompt = self._build_user_prompt(context, query)

            logger.info(f"🤖 Streaming response using model: {model}")

            # Stream through proxy
            stream = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={"response": full_response, "length": len(full_response)}
                )

        except Exception as e:
            logger.error(f"❌ Error in stream response: {e}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )

            yield f"\n\n[Error: {str(e)}]"

    @observe(name="validate_context_relevance")
    def validate_context_relevance(self, context: str, query: str) -> Dict[str, Any]:
        """Check if context is relevant to the query"""
        context_lower = context.lower()
        query_words = set(query.lower().split())

        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are',
                      'was', 'were'}
        query_words = query_words - stop_words

        overlap_count = sum(1 for word in query_words if word in context_lower)
        relevance_score = overlap_count / len(query_words) if query_words else 0

        result = {
            "is_relevant": relevance_score > 0.2,
            "relevance_score": round(relevance_score, 2),
            "matched_keywords": overlap_count,
            "total_keywords": len(query_words)
        }

        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                input={"context_length": len(context), "query": query},
                output=result
            )

        return result


# Convenience function
@observe(name="quick_generate")
def quick_generate(
        context: str,
        query: str,
        model: str = "llama-3.3-70b-groq",
        base_url: str = "http://localhost:4000/v1"
) -> str:
    """Quick generation function"""
    generator = Generator(base_url=base_url)
    result = generator.generate_response(context, query, model)
    return result["response"]
