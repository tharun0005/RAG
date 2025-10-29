import logging
import os
import warnings
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*getaddrinfo failed.*")

try:
    from langfuse import Langfuse

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        timeout=60,
        max_retries=3,
        flush_at=1,
        flush_interval=5
    )
    LANGFUSE_ENABLED = True

    logging.getLogger("langfuse").setLevel(logging.ERROR)

    logger.info("LangFuse initialized for Generator")
except Exception as e:
    LANGFUSE_ENABLED = False
    logger.warning(f"LangFuse not initialized: {e}")


class Generator:
    """Enhanced generator for RAG QA using LiteLLM proxy with Guardrails"""

    def __init__(
            self,
            base_url: str = "http://localhost:4000/v1",
            api_key: str = "dummy-key",
            enable_guardrails: bool = True
    ):
        """
        Initialize generator with LiteLLM proxy

        Args:
            base_url: LiteLLM proxy URL (default: http://localhost:4000/v1)
            api_key: API key (can be dummy for local proxy)
            enable_guardrails: Enable guardrails for all requests
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.enable_guardrails = enable_guardrails
        self.guardrails = ["sensitive-topics-guard", "toxic-language-guard"]

        logger.info(f"Generator initialized with LiteLLM proxy at: {base_url}")
        if self.enable_guardrails:
            logger.info(f"Guardrails enabled: {', '.join(self.guardrails)}")

    @observe(name="generate_response", capture_input=True, capture_output=True)
    def generate_response(
            self,
            context: str,
            query: str,
            model: str = "llama-3.3-70b-groq",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            system_prompt: Optional[str] = None,
            enable_guardrails: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate QA response with strict context adherence and guardrails protection

        Args:
            context: Retrieved context for answering the query
            query: User's question
            model: Model name from config.yaml
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (optional)
            enable_guardrails: Override instance-level guardrails setting
        """
        try:
            if LANGFUSE_ENABLED:
                langfuse_context.update_current_trace(
                    tags=["generation", "rag", model],
                    metadata={
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "guardrails_enabled": enable_guardrails if enable_guardrails is not None else self.enable_guardrails
                    }
                )

            if not context or not context.strip():
                logger.warning("Empty context provided")
                return {
                    "response": "I don't have any relevant information in my knowledge base to answer that question.",
                    "model_used": model,
                    "has_context": False,
                    "guardrails_passed": False
                }

            if not query or not query.strip():
                logger.warning("Empty query provided")
                return {
                    "response": "Please provide a question.",
                    "model_used": model,
                    "has_context": bool(context),
                    "guardrails_passed": False
                }

            if not system_prompt:
                system_prompt = self._build_system_prompt()

            user_prompt = self._build_user_prompt(context, query)

            logger.info(f"Generating response using model: {model}")

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    input={
                        "query": query,
                        "context_length": len(context),
                        "system_prompt_length": len(system_prompt)
                    }
                )

            use_guardrails = enable_guardrails if enable_guardrails is not None else self.enable_guardrails

            extra_body = None
            if use_guardrails:
                logger.info(f"Guardrails will run automatically: {', '.join(self.guardrails)}")

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            response_content = response.choices[0].message.content.strip()

            is_hallucination = self._is_hallucination(response_content)
            if is_hallucination:
                logger.warning("Potential hallucination detected")
                response_content = "I don't have enough information in the provided context to answer that question accurately."

            usage = response.usage if hasattr(response, 'usage') else None
            tokens_used = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            result = {
                "response": response_content,
                "model_used": model,
                "has_context": True,
                "hallucination_detected": is_hallucination,
                "guardrails_passed": True,
                "guardrails_used": self.guardrails if use_guardrails else [],
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

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    output={
                        "response": response_content,
                        "response_length": len(response_content),
                        "hallucination_detected": is_hallucination,
                        "guardrails_passed": True
                    },
                    metadata={
                        "tokens_used": tokens_used,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "guardrails_used": self.guardrails if use_guardrails else []
                    }
                )

            logger.info(f"Generated response ({len(response_content)} chars, {tokens_used} tokens)")
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating response: {error_msg}", exc_info=True)

            guardrails_blocked = any(keyword in error_msg.lower() for keyword in
                                     ["guardrail", "toxic", "sensitive", "blocked", "violated", "validation"])

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error_msg,
                    metadata={"guardrails_blocked": guardrails_blocked}
                )

            if guardrails_blocked:
                return {
                    "response": "I apologize, but I cannot process this request as it violates our content policy.",
                    "model_used": model,
                    "error": error_msg,
                    "has_context": bool(context),
                    "guardrails_passed": False,
                    "guardrails_blocked": True
                }
            else:
                return {
                    "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                    "model_used": model,
                    "error": error_msg,
                    "has_context": bool(context),
                    "guardrails_passed": False
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
- "I don't have enough information in the provided context to answer that question."
- "According to the context, [specific answer from context]."
- "The context mentions [X] but doesn't provide information about [Y]."

**Examples of INCORRECT responses:**
- Answering from general knowledge when context doesn't have the answer
- Making assumptions beyond what's explicitly stated
- Providing information not found in the context"""

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
            system_prompt: Optional[str] = None,
            enable_guardrails: Optional[bool] = None
    ):
        """Stream QA response from LLM with guardrails protection"""
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

            logger.info(f"Streaming response using model: {model}")

            use_guardrails = enable_guardrails if enable_guardrails is not None else self.enable_guardrails
            if use_guardrails:
                logger.info("Pre-call guardrails will run automatically for streaming")

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
            error_msg = str(e)
            logger.error(f"Error in stream response: {error_msg}")

            guardrails_blocked = any(keyword in error_msg.lower() for keyword in
                                     ["guardrail", "toxic", "sensitive", "blocked", "violated", "validation"])

            if LANGFUSE_ENABLED:
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error_msg,
                    metadata={"guardrails_blocked": guardrails_blocked}
                )

            if guardrails_blocked:
                yield "\n\n[Request blocked by content policy]"
            else:
                yield f"\n\n[Error: {error_msg}]"

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


@observe(name="quick_generate")
def quick_generate(
        context: str,
        query: str,
        model: str = "llama-3.3-70b-groq",
        base_url: str = "http://localhost:4000/v1",
        enable_guardrails: bool = True
) -> str:
    """Quick generation function with guardrails"""
    generator = Generator(base_url=base_url, enable_guardrails=enable_guardrails)
    result = generator.generate_response(context, query, model)
    return result["response"]
