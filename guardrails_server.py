from flask import Flask, request, jsonify
from guardrails import Guard
from guardrails.hub import ToxicLanguage
import logging
import json
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

logger.info("Initializing guardrails...")

try:
    sensitive_topic_guard = Guard(name='sensitive-topic-guard')
    sensitive_topic_guard.use(
        ToxicLanguage,
        threshold=0.5,
        validation_method="sentence",
        on_fail="exception"
    )

    toxic_language_guard = Guard(name='toxic-language-guard')
    toxic_language_guard.use(
        ToxicLanguage,
        threshold=0.6,
        validation_method="sentence",
        on_fail="exception"
    )

    combined_guard = Guard(name='combined-safety-guard')
    combined_guard.use(
        ToxicLanguage,
        threshold=0.5,
        validation_method="sentence",
        on_fail="exception"
    )

    guards = {
        'sensitive-topic-guard': sensitive_topic_guard,
        'toxic-language-guard': toxic_language_guard,
        'combined-safety-guard': combined_guard
    }

    logger.info(f"Successfully initialized {len(guards)} guard(s)")

except Exception as e:
    logger.error(f"Failed to initialize guards: {e}")
    guards = {}


@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "guards": list(guards.keys()),
        "guard_count": len(guards)
    })


@app.route('/guards', methods=['GET'])
def list_guards():
    return jsonify(list(guards.keys()))


@app.route('/guards/<guard_name>/validate', methods=['POST'])
def validate_litellm(guard_name):
    """LiteLLM validation endpoint with proper JSON response"""
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({
                "validation_passed": False,
                "error": "No JSON data provided"
            }), 400

        if guard_name not in guards:
            logger.warning(f"Guard '{guard_name}' not found")
            return jsonify({
                "validation_passed": False,
                "error": f"Guard '{guard_name}' not found",
                "available_guards": list(guards.keys())
            }), 404

        text = None

        if 'llmOutput' in data:
            text = data['llmOutput']
        elif 'text' in data:
            text = data['text']
        elif 'prompt' in data:
            text = data['prompt']
        elif 'messages' in data:
            messages = data['messages']
            if isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and 'content' in last_message:
                    text = last_message['content']

        MAX_LENGTH = 5000
        if text and len(text) > MAX_LENGTH:
            logger.warning(f"Input text too long ({len(text)} chars), truncating to {MAX_LENGTH}")
            text = text[:MAX_LENGTH]

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text provided: {type(text)}")
            return jsonify({
                "validation_passed": False,
                "error": "No valid text provided in payload"
            }), 400

        guard = guards[guard_name]

        logger.info(f"Validating with '{guard_name}': {text[:100]}...")

        try:
            result = guard.validate(text)

            validated_text = text
            if hasattr(result, 'validated_output'):
                validated_text = result.validated_output
            elif hasattr(result, 'raw_llm_output'):
                validated_text = result.raw_llm_output

            logger.info(f"Validation PASSED with '{guard_name}'")
            return jsonify({
                "validation_passed": True,
                "validated_output": validated_text,
                "raw_llm_output": text
            }), 200

        except Exception as validation_error:
            error_message = str(validation_error)
            logger.warning(f"Validation FAILED with '{guard_name}': {error_message[:200]}")

            return jsonify({
                "validation_passed": False,
                "error": error_message,
                "guard_name": guard_name,
                "error_type": "validation_failed"
            }), 200

    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in validate_litellm: {error_message}")
        logger.error(f"Full traceback: {error_trace}")

        return jsonify({
            "validation_passed": False,
            "error": error_message,
            "guard_name": guard_name,
            "error_type": "server_error"
        }), 500


@app.route('/validate', methods=['POST'])
def validate():
    """Generic validation endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        guard_name = data.get('guard_name', 'combined-safety-guard')
        text = data.get('text', data.get('llmOutput', ''))

        MAX_LENGTH = 5000
        if text and len(text) > MAX_LENGTH:
            logger.warning(f"Input text too long ({len(text)} chars), truncating to {MAX_LENGTH}")
            text = text[:MAX_LENGTH]

        if not text or not isinstance(text, str):
            return jsonify({"error": "No valid text provided"}), 400

        if guard_name not in guards:
            return jsonify({
                "error": f"Guard '{guard_name}' not found",
                "available_guards": list(guards.keys())
            }), 404

        guard = guards[guard_name]

        try:
            result = guard.validate(text)

            validated_text = text
            if hasattr(result, 'validated_output'):
                validated_text = result.validated_output
            elif hasattr(result, 'raw_llm_output'):
                validated_text = result.raw_llm_output

            logger.info(f"Generic validation PASSED with '{guard_name}'")
            return jsonify({
                "validated": True,
                "guard_name": guard_name,
                "result": validated_text
            }), 200

        except Exception as validation_error:
            error_message = str(validation_error)
            logger.warning(f"Generic validation FAILED: {error_message[:200]}")

            return jsonify({
                "validated": False,
                "error": error_message,
                "guard_name": guard_name,
                "error_type": "validation_failed"
            }), 200

    except Exception as e:
        logger.error(f"Unexpected error in validate: {str(e)}", exc_info=True)
        return jsonify({
            "validated": False,
            "error": str(e),
            "error_type": "server_error"
        }), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Guardrails AI Server",
        "version": "1.0.0",
        "status": "running",
        "available_guards": list(guards.keys()),
        "endpoints": {
            "health_check": "/health-check",
            "list_guards": "/guards",
            "litellm_validate": "/guards/<guard_name>/validate",
            "generic_validate": "/validate"
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Guardrails AI Server")
    print("=" * 60)
    print(f"Guards: {', '.join(guards.keys())}")
    print("Server: http://0.0.0.0:8001")
    print("Endpoints:")
    print("  - GET  /health-check")
    print("  - GET  /guards")
    print("  - POST /guards/<guard_name>/validate (LiteLLM)")
    print("  - POST /validate (Generic)")
    print("=" * 60 + "\n")

    from werkzeug.serving import WSGIRequestHandler

    WSGIRequestHandler.timeout = 60

    app.run(host='0.0.0.0', port=8001, debug=False, threaded=True)
