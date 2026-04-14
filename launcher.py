"""
SHARKFLOW v5.0 — Polymarket Intelligence
Desktop Launcher via PyWebView
Developed by: Carlos David Donoso Cordero (ddchack)
"""

import os
import sys
import time
import signal
import subprocess
import threading
import requests
from pathlib import Path

# Directorio raiz del proyecto
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_URL = "http://localhost:8888"
STATUS_URL = f"{BACKEND_URL}/api/status"
STARTUP_TIMEOUT = 60  # segundos (numpy/scipy tardan en cargar)

# ══════════════════════════════════════════════════════
# Logo ASCII para splash screen en terminal
# ══════════════════════════════════════════════════════
SPLASH = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗██╗  ██╗ █████╗ ██████╗ ██╗  ██╗███████╗██╗       ║
║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝██║       ║
║   ███████╗███████║███████║██████╔╝█████╔╝ █████╗  ██║       ║
║   ╚════██║██╔══██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝  ██║       ║
║   ███████║██║  ██║██║  ██║██║  ██║██║  ██╗██║     ███████╗  ║
║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝  ║
║                                                              ║
║       SHARKFLOW v5.0 — Polymarket Intelligence               ║
║       by Carlos David Donoso Cordero (ddchack)               ║
║                                                              ║
║   KL Divergence | Multi-Kelly | ELO | Poisson | Bayesian     ║
║   HMM Regime | Monte Carlo | LMSR | Extremization | VPIN    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_env():
    """Cargar variables de entorno desde .env"""
    try:
        from dotenv import load_dotenv
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print("[OK] Variables de entorno cargadas desde .env")
        else:
            print("[WARN] Archivo .env no encontrado, usando variables del sistema")
    except ImportError:
        print("[WARN] python-dotenv no instalado, usando variables del sistema")


def kill_port(port: int):
    """Matar cualquier proceso que esté usando el puerto dado."""
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True)
        # Decodificar con cp1252 (encoding nativo de Windows en cmd)
        stdout = result.stdout.decode("cp1252", errors="replace")
        for line in stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid > 0:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True)
                    print(f"[OK] Puerto {port} liberado (PID {pid} terminado)")
                    time.sleep(1)
                break
    except Exception:
        pass


def start_backend():
    """Iniciar el backend FastAPI como subproceso."""
    env = os.environ.copy()
    # Forzar UTF-8 para evitar UnicodeEncodeError con caracteres especiales en la terminal
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "api_server.py"],
        cwd=str(BACKEND_DIR),
        env=env,
        # No capturar stdout — dejar que el backend imprima en la terminal para ver errores
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return process


def wait_for_server(timeout=STARTUP_TIMEOUT):
    """Esperar a que el servidor responda en /api/status."""
    start = time.time()
    dots = 0
    while time.time() - start < timeout:
        try:
            resp = requests.get(STATUS_URL, timeout=2)
            if resp.status_code == 200:
                print("\n[OK] Backend listo!")
                return True
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        dots += 1
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        sym = spinner[dots % len(spinner)]
        elapsed = int(time.time() - start)
        print(f"\r  {sym} Iniciando backend... ({elapsed}s)", end="", flush=True)
        time.sleep(0.5)

    print(f"\n[ERROR] El backend no respondio despues de {timeout} segundos.")
    return False


def cleanup_backend(process):
    """Terminar el proceso del backend limpiamente."""
    if process and process.poll() is None:
        print("[...] Cerrando backend...")
        try:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
            print("[OK] Backend cerrado correctamente.")
        except subprocess.TimeoutExpired:
            print("[WARN] Forzando cierre del backend...")
            process.kill()
            process.wait()
            print("[OK] Backend terminado.")
        except Exception as e:
            print(f"[ERROR] Error cerrando backend: {e}")
            process.kill()


def main():
    print(SPLASH)
    print("  Iniciando SHARKFLOW Desktop...\n")

    # 1. Cargar entorno
    load_env()

    # 2. Liberar puerto y arrancar backend
    print("[...] Verificando puerto 8888...")
    kill_port(8888)
    print("[...] Iniciando servidor backend (FastAPI/Uvicorn)...")
    backend_process = start_backend()

    # 3. Esperar a que el servidor este listo
    if not wait_for_server():
        print("\n" + "=" * 60)
        print("  ERROR: No se pudo iniciar el backend.")
        print("  Verifica que las dependencias esten instaladas:")
        print("    pip install -r requirements.txt")
        print("  Y que el puerto 8888 no este en uso.")
        print("=" * 60)
        cleanup_backend(backend_process)
        sys.exit(1)

    # 4. Abrir ventana de escritorio en modo app (sin barra del navegador)
    print("[OK] Abriendo ventana de escritorio...")
    app_launched = False

    # Google Chrome primero, Edge como fallback
    browser_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # Fallback: Microsoft Edge
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    app_flags = [
        f"--app={BACKEND_URL}/app?v={int(time.time())}",
        "--window-size=1400,900",
        "--window-position=80,40",
        "--disable-extensions",
        "--no-first-run",
        "--disable-default-browser-check",
        "--disable-http-cache",
        "--disk-cache-size=1",
    ]

    for browser_path in browser_candidates:
        if os.path.exists(browser_path):
            print(f"[OK] Usando: {os.path.basename(browser_path)}")
            try:
                subprocess.Popen([browser_path] + app_flags)
                app_launched = True
                print(f"\n  Dashboard abierto en Chrome.")
                print("  Presiona Ctrl+C aqui para cerrar el servidor.\n")
                # NO esperar al proceso del browser — Chrome ya abierto pasa el URL
                # a la instancia existente y el subproceso termina inmediatamente.
                # Esperar al BACKEND o a Ctrl+C.
                try:
                    backend_process.wait()
                except KeyboardInterrupt:
                    pass
                break
            except Exception as e:
                print(f"[WARN] No se pudo abrir {browser_path}: {e}")

    if not app_launched:
        # Fallback: abrir en el navegador por defecto
        import webbrowser
        print("[WARN] No se encontro Edge ni Chrome. Abriendo en navegador por defecto...")
        webbrowser.open(BACKEND_URL)
        print(f"\n  Dashboard en: {BACKEND_URL}")
        print("  Presiona Ctrl+C para cerrar el servidor.\n")
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            pass

    # 5. Al cerrar la ventana, terminar el backend
    cleanup_backend(backend_process)
    print("\n  SHARKFLOW cerrado. Hasta pronto!\n")


if __name__ == "__main__":
    main()
