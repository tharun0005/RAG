import os
import subprocess
import time
import signal
import sys
from threading import Thread
import requests
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"


def load_environment():
    """Load environment variables and verify required ones"""
    print("Loading environment variables...")

    if not load_dotenv():
        print("WARNING: .env file not found, using system environment variables")

    required_vars = ["GROQ_API_KEY"]
    optional_vars = ["HF_TOKEN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]

    print("Environment Variables Status:")

    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   [OK] {var}: {masked_value}")
        else:
            print(f"   [ERROR] {var}: NOT SET")
            missing_required.append(var)

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   [OPTIONAL] {var}: {masked_value}")
        else:
            print(f"   [WARNING] {var}: Not set (optional)")

    if missing_required:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing_required)}")
        return False

    print("[SUCCESS] Environment variables loaded successfully\n")
    return True


def test_groq_api_key():
    """Test if the Groq API key is valid"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[ERROR] No GROQ_API_KEY found to test")
        return False

    print("Testing Groq API key...")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say 'Hello World'"}],
                "max_tokens": 10
            },
            timeout=10
        )

        if response.status_code == 200:
            print("[OK] Groq API key is valid!\n")
            return True
        else:
            print(f"[FAIL] Groq API key test failed: {response.status_code}\n")
            return False

    except Exception as e:
        print(f"[ERROR] Error testing Groq API key: {e}\n")
        return False


def check_config_file():
    """Verify config.yaml exists and is valid"""
    if not os.path.exists("config.yaml"):
        print("[ERROR] config.yaml not found!")
        return False

    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        if not config.get("model_list"):
            print("[WARNING] No models found in config.yaml")
            return False

        model_count = len(config["model_list"])
        print(f"[OK] Found {model_count} model(s) in config.yaml")

        print("     Configured models:")
        for model in config["model_list"]:
            model_name = model.get("model_name", "Unknown")
            provider = model.get("model_info", {}).get("provider", "Unknown")
            print(f"       - {model_name} ({provider})")

        if config.get("guardrails"):
            guardrails = config["guardrails"]
            guardrail_count = len(guardrails)
            print(f"[OK] Found {guardrail_count} guardrail(s) configured")
            print("     Configured guardrails:")
            for guardrail in guardrails:
                name = guardrail.get("guardrail_name", "Unknown")
                mode = guardrail.get("litellm_params", {}).get("mode", "unknown")
                default_on = guardrail.get("litellm_params", {}).get("default_on", False)
                status = "ACTIVE" if default_on else "INACTIVE"
                print(f"       - {name} (mode: {mode}) [{status}]")
        else:
            print("[WARNING] No guardrails configured in config.yaml")

        return True

    except Exception as e:
        print(f"[ERROR] Error reading config.yaml: {e}")
        return False


def check_guardrails_server():
    """Verify guardrails_server.py exists"""
    if not os.path.exists("guardrails_server.py"):
        print("[WARNING] guardrails_server.py not found!")
        print("          Guardrails functionality will not be available")
        return False

    if not os.path.exists("config.py"):
        print("[WARNING] config.py not found!")
        print("          Guardrails configuration file missing")
        return False

    print("[OK] guardrails_server.py found")
    print("[OK] config.py found")
    return True


def start_guardrails():
    """Start Guardrails Flask server on port 8001"""
    print("\n" + "=" * 60)
    print("Starting Guardrails AI server...")
    print("=" * 60)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            [sys.executable, "guardrails_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env
        )
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start Guardrails: {e}")
        return None


def start_litellm():
    """Start LiteLLM proxy server"""
    print("\n" + "=" * 60)
    print("Starting LiteLLM proxy server...")
    print("=" * 60)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["LITELLM_LOG"] = "INFO"

        process = subprocess.Popen(
            ["litellm", "--config", "config.yaml", "--port", "4000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env
        )
        return process
    except FileNotFoundError:
        print("[ERROR] LiteLLM command not found. Is it installed?")
        print("        Install with: pip install 'litellm[proxy]'")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to start LiteLLM: {e}")
        return None


def start_fastapi():
    """Start FastAPI server"""
    print("\n" + "=" * 60)
    print("Starting FastAPI server...")
    print("=" * 60)

    if not os.path.exists("main.py"):
        print("[ERROR] main.py not found!")
        return None

    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start FastAPI: {e}")
        return None


def print_output(pipe, process_name):
    """Print output from a process pipe"""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(f"[{process_name}] {line}", end='')
    except Exception as e:
        print(f"[{process_name}] Error: {e}")
    finally:
        pipe.close()


def check_guardrails_health():
    """Check if Guardrails server is healthy"""
    try:
        response = requests.get("http://localhost:8001/health-check", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"        Guardrails status: {data.get('status')}")
            guards = data.get('guards', [])
            print(f"        Loaded {len(guards)} guard(s): {', '.join(guards)}")
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def check_litellm_health():
    """Check if LiteLLM is healthy"""
    try:
        response = requests.get("http://localhost:4000/health", timeout=10)
        if response.status_code == 200:
            models_response = requests.get("http://localhost:4000/models", timeout=5)
            if models_response.status_code == 200:
                models = models_response.json()
                model_count = len(models.get("data", []))
                print(f"        LiteLLM loaded {model_count} model(s)")
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def check_fastapi_health():
    """Check if FastAPI is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"        FastAPI status: {health_data.get('status')}")
            print(f"        RAG available: {health_data.get('rag_available', False)}")
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def test_guardrails_integration():
    """Test if LiteLLM can reach Guardrails server"""
    try:
        print("\n[TEST] Testing Guardrails integration...")
        response = requests.post(
            "http://localhost:8001/guards/sensitive-topic-guard/validate",
            json={"llmOutput": "Hello world test"},
            timeout=5
        )
        if response.status_code in [200, 400]:
            print("        Guardrails integration: OK")
            return True
        else:
            print(f"        Guardrails integration: WARN (status {response.status_code})")
            return False
    except Exception as e:
        print(f"        Guardrails integration: FAIL ({e})")
        return False


def terminate_process(process, name, timeout=5):
    """Safely terminate a process"""
    if process and process.poll() is None:
        print(f"[STOP] Stopping {name}...")

        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()

        try:
            process.wait(timeout=timeout)
            print(f"[OK] {name} stopped")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print(f"[OK] {name} killed")


def main():
    print("\n" + "=" * 70)
    print("AI RAG Dashboard - Server Manager")
    print("=" * 70 + "\n")

    if not load_environment():
        sys.exit(1)

    print("")
    if not check_config_file():
        sys.exit(1)

    print("")
    guardrails_available = check_guardrails_server()

    print("")
    if not test_groq_api_key():
        print("[WARNING] Groq API key test failed!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    guardrails_process = None
    litellm_process = None
    fastapi_process = None

    try:
        if guardrails_available:
            guardrails_process = start_guardrails()
            if guardrails_process:
                Thread(target=print_output, args=(guardrails_process.stdout, "Guardrails"), daemon=True).start()
                Thread(target=print_output, args=(guardrails_process.stderr, "Guardrails-ERR"), daemon=True).start()

                print("\n[WAIT] Waiting for Guardrails...")
                for i in range(15):
                    if check_guardrails_health():
                        print("[OK] Guardrails ready")
                        test_guardrails_integration()
                        print()
                        break
                    elif i == 14:
                        print("[WARNING] Continuing without Guardrails\n")
                        terminate_process(guardrails_process, "Guardrails")
                        guardrails_process = None
                    else:
                        if guardrails_process.poll() is not None:
                            print("[WARNING] Guardrails died\n")
                            guardrails_process = None
                            break
                        time.sleep(2)

        litellm_process = start_litellm()
        if not litellm_process:
            sys.exit(1)

        Thread(target=print_output, args=(litellm_process.stdout, "LiteLLM"), daemon=True).start()
        Thread(target=print_output, args=(litellm_process.stderr, "LiteLLM-ERR"), daemon=True).start()

        print("\n[WAIT] Waiting for LiteLLM...")
        for i in range(30):
            if check_litellm_health():
                print("[OK] LiteLLM ready\n")
                break
            elif i == 29:
                print("[ERROR] LiteLLM failed to start")
                terminate_process(guardrails_process, "Guardrails")
                terminate_process(litellm_process, "LiteLLM")
                sys.exit(1)
            else:
                time.sleep(2)

        fastapi_process = start_fastapi()
        if not fastapi_process:
            terminate_process(guardrails_process, "Guardrails")
            terminate_process(litellm_process, "LiteLLM")
            sys.exit(1)

        Thread(target=print_output, args=(fastapi_process.stdout, "FastAPI"), daemon=True).start()
        Thread(target=print_output, args=(fastapi_process.stderr, "FastAPI-ERR"), daemon=True).start()

        print("\n[WAIT] Waiting for FastAPI...")
        for i in range(15):
            if check_fastapi_health():
                print("[OK] FastAPI ready\n")
                break
            else:
                time.sleep(2)

        print("=" * 70)
        print("SUCCESS: All servers running!")
        print("=" * 70)
        print("")
        if guardrails_process:
            print("   Guardrails:  http://localhost:8001  Protected")
        else:
            print("   Guardrails:  Not running  Unprotected")
        print("   LiteLLM:     http://localhost:4000")
        print("   Dashboard:   http://localhost:8000")
        print("")
        print("Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        while True:
            if litellm_process.poll() is not None:
                print("\n[ERROR] LiteLLM died")
                break
            if fastapi_process.poll() is not None:
                print("\n[ERROR] FastAPI died")
                break
            if guardrails_process and guardrails_process.poll() is not None:
                print("\n[WARNING] Guardrails died - continuing without protection")
                guardrails_process = None
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Shutting down...")
    finally:
        print("\n" + "=" * 70)
        print("Shutting down...")
        print("=" * 70)
        terminate_process(fastapi_process, "FastAPI")
        terminate_process(litellm_process, "LiteLLM")
        terminate_process(guardrails_process, "Guardrails")
        print("\n[DONE] Stopped")


if __name__ == "__main__":
    main()
