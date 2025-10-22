import os
import subprocess
import time
import signal
import sys
from threading import Thread
import requests
from dotenv import load_dotenv

# Fix for Windows Unicode encoding issues
if sys.platform == "win32":
    # Set UTF-8 encoding for Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"


def load_environment():
    """Load environment variables and verify required ones"""
    print("Loading environment variables...")

    # Load .env file
    if not load_dotenv():
        print("WARNING: .env file not found, using system environment variables")

    # Check if .env exists but couldn't be loaded
    if not os.path.exists('.env'):
        print("WARNING: .env file not found. Using system environment variables.")

    # Debug: Show what environment variables are loaded
    required_vars = ["GROQ_API_KEY"]
    optional_vars = ["HF_TOKEN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]

    print("Environment Variables Status:")

    # Check required variables
    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first and last few characters for security
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   [OK] {var}: {masked_value}")
        else:
            print(f"   [ERROR] {var}: NOT SET")
            missing_required.append(var)

    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   [OPTIONAL] {var}: {masked_value}")
        else:
            print(f"   [WARNING] {var}: Not set (optional)")

    # ✅ Validate LangFuse host URL format
    langfuse_host = os.getenv("LANGFUSE_HOST")
    if langfuse_host:
        if not (langfuse_host.startswith('http://') or langfuse_host.startswith('https://')):
            print(f"[WARNING] LANGFUSE_HOST should start with http:// or https://")
            print(f"          Current value: {langfuse_host}")
        if "..." in langfuse_host or langfuse_host.endswith('.com') and len(langfuse_host) < 20:
            print(f"[ERROR] LANGFUSE_HOST appears to be incomplete or invalid")
            print(f"        Expected: https://cloud.langfuse.com")
            print(f"        Current: {langfuse_host}")

    # Validate Groq API key format
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        if not groq_key.startswith('gsk_'):
            print("[WARNING] GROQ_API_KEY doesn't start with 'gsk_' - may be invalid")
        if len(groq_key) < 20:
            print("[WARNING] GROQ_API_KEY seems too short - may be invalid")

    if missing_required:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing_required)}")
        print("Please check your .env file or set these environment variables")
        return False

    print("[SUCCESS] Environment variables loaded successfully\n")
    return True


def test_groq_api_key():
    """Test if the Groq API key is valid by making a direct API call"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[ERROR] No GROQ_API_KEY found to test")
        return False

    print("Testing Groq API key...")
    try:
        # Test directly with Groq API
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
        elif response.status_code == 401:
            print(f"[FAIL] Groq API key is INVALID (Unauthorized)\n")
            return False
        else:
            print(f"[FAIL] Groq API key test failed: {response.status_code} - {response.text[:100]}\n")
            return False

    except Exception as e:
        print(f"[ERROR] Error testing Groq API key: {e}\n")
        return False


def check_config_file():
    """Verify config.yaml exists and is valid"""
    if not os.path.exists("config.yaml"):
        print("[ERROR] config.yaml not found!")
        print("        Please create config.yaml before starting the server")
        return False

    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Check if model_list exists and has at least one model
        if not config.get("model_list"):
            print("[WARNING] No models found in config.yaml")
            return False

        model_count = len(config["model_list"])
        print(f"[OK] Found {model_count} model(s) in config.yaml")

        # List configured models
        print("     Configured models:")
        for model in config["model_list"]:
            model_name = model.get("model_name", "Unknown")
            provider = model.get("model_info", {}).get("provider", "Unknown")
            print(f"       - {model_name} ({provider})")

        return True

    except Exception as e:
        print(f"[ERROR] Error reading config.yaml: {e}")
        return False


def start_litellm():
    """Start LiteLLM proxy server"""
    print("\n" + "=" * 60)
    print("Starting LiteLLM proxy server...")
    print("=" * 60)

    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["LITELLM_LOG"] = "INFO"
        env["LITELLM_DISABLE_BANNER"] = "True"

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

    time.sleep(3)  # Wait for LiteLLM to start

    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("[ERROR] main.py not found!")
        return None

    try:
        env = os.environ.copy()

        process = subprocess.Popen(
            [sys.executable, "main.py"],
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
        print(f"[ERROR] Failed to start FastAPI: {e}")
        return None


def print_output(pipe, process_name):
    """Print output from a process pipe with filtering"""
    try:
        # ✅ Filter out noisy logs
        noise_filters = [
            "Generating embeddings:",  # HuggingFace progress bars
            "INFO:     127.0.0.1:",  # FastAPI request logs (optional)
        ]

        for line in iter(pipe.readline, ''):
            if line:
                # Check if line should be filtered
                should_print = True
                for noise in noise_filters:
                    if noise in line:
                        should_print = False
                        break

                if should_print:
                    print(f"[{process_name}] {line}", end='')
    except Exception as e:
        print(f"[{process_name}] Error reading output: {e}")
    finally:
        pipe.close()


def check_litellm_health():
    """Check if LiteLLM is healthy by making a request to /health"""
    try:
        response = requests.get("http://localhost:4000/health", timeout=10)
        if response.status_code == 200:
            # Also check if models are loaded
            try:
                models_response = requests.get("http://localhost:4000/models", timeout=5)
                if models_response.status_code == 200:
                    models = models_response.json()
                    model_count = len(models.get("data", []))
                    print(f"        LiteLLM loaded {model_count} model(s)")
                    return True
            except:
                return True  # Health endpoint passed, that's enough
        return False
    except requests.exceptions.RequestException:
        return False


def check_fastapi_health():
    """Check if FastAPI is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"        FastAPI status: {health_data.get('status', 'unknown')}")
            print(f"        RAG available: {health_data.get('rag_available', False)}")
            print(f"        LangFuse enabled: {health_data.get('langfuse_enabled', False)}")
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def terminate_process(process, name, timeout=5):
    """Safely terminate a process"""
    if process and process.poll() is None:
        print(f"[STOP] Stopping {name}...")

        # Try graceful shutdown first
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()

        try:
            process.wait(timeout=timeout)
            print(f"[OK] {name} stopped gracefully")
        except subprocess.TimeoutExpired:
            print(f"[WARN] {name} didn't terminate gracefully, forcing...")
            process.kill()
            process.wait()
            print(f"[OK] {name} killed")


def main():
    print("\n" + "=" * 70)
    print("AI RAG Dashboard - Server Manager")
    print("=" * 70 + "\n")

    # Step 1: Load environment variables
    if not load_environment():
        sys.exit(1)

    # ✅ Step 1.5: Check config file
    print("")
    if not check_config_file():
        print("[ERROR] config.yaml validation failed")
        sys.exit(1)

    # Step 2: Test Groq API key (optional but recommended)
    print("")
    if not test_groq_api_key():
        print("[WARNING] Groq API key test failed!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    litellm_process = None
    fastapi_process = None

    try:
        # Step 3: Start LiteLLM first
        litellm_process = start_litellm()
        if not litellm_process:
            sys.exit(1)

        # Start output threads
        Thread(target=print_output, args=(litellm_process.stdout, "LiteLLM"), daemon=True).start()
        Thread(target=print_output, args=(litellm_process.stderr, "LiteLLM-ERR"), daemon=True).start()

        # Wait for LiteLLM to be ready
        print("\n[WAIT] Waiting for LiteLLM to start...")
        max_retries = 30
        for i in range(max_retries):
            if check_litellm_health():
                print("[OK] LiteLLM is ready!\n")
                break
            elif i == max_retries - 1:
                print("[ERROR] LiteLLM failed to start within 60 seconds")
                print("        Check the logs above for errors")
                terminate_process(litellm_process, "LiteLLM")
                sys.exit(1)
            else:
                if litellm_process.poll() is not None:
                    print("[ERROR] LiteLLM process died during startup")
                    print(f"        Exit code: {litellm_process.returncode}")
                    sys.exit(1)
                print(f"        Attempt {i + 1}/{max_retries}...")
                time.sleep(2)

        # Step 4: Then start FastAPI
        fastapi_process = start_fastapi()
        if not fastapi_process:
            print("[ERROR] Failed to start FastAPI")
            terminate_process(litellm_process, "LiteLLM")
            sys.exit(1)

        # Start output threads for FastAPI
        Thread(target=print_output, args=(fastapi_process.stdout, "FastAPI"), daemon=True).start()
        Thread(target=print_output, args=(fastapi_process.stderr, "FastAPI-ERR"), daemon=True).start()

        # Wait for FastAPI to be ready
        print("\n[WAIT] Waiting for FastAPI to start...")
        for i in range(15):
            if check_fastapi_health():
                print("[OK] FastAPI is ready!\n")
                break
            elif i == 14:
                print("[WARNING] FastAPI might not be fully ready, but continuing...\n")
            else:
                if fastapi_process.poll() is not None:
                    print("[ERROR] FastAPI process died during startup")
                    print(f"        Exit code: {fastapi_process.returncode}")
                    terminate_process(litellm_process, "LiteLLM")
                    sys.exit(1)
                time.sleep(2)

        print("=" * 70)
        print("✅ SUCCESS: Both servers are running!")
        print("=" * 70)
        print("")
        print("📌 Access URLs:")
        print("   🌐 Dashboard:     http://localhost:8000")
        print("   🔧 Data Ingestion: http://localhost:8000/data-ingestion")
        print("   🤖 LiteLLM API:    http://localhost:4000")
        print("   📊 LiteLLM UI:     http://localhost:4000/ui")
        print("")
        print("⌨️  Press Ctrl+C to stop both servers")
        print("=" * 70 + "\n")

        # Monitor processes
        while True:
            if litellm_process.poll() is not None:
                print("\n[ERROR] LiteLLM process died unexpectedly")
                print(f"        Exit code: {litellm_process.returncode}")
                break
            if fastapi_process.poll() is not None:
                print("\n[ERROR] FastAPI process died unexpectedly")
                print(f"        Exit code: {fastapi_process.returncode}")
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received interrupt signal (Ctrl+C)...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("Shutting down servers...")
        print("=" * 70)
        terminate_process(fastapi_process, "FastAPI")
        terminate_process(litellm_process, "LiteLLM")
        print("\n[DONE] Servers stopped successfully.")
        print("=" * 70)


if __name__ == "__main__":
    main()
