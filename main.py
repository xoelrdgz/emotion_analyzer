"""Application entry point for the Emotion Analyzer Streamlit interface."""

import sys
import subprocess
from pathlib import Path

def run_streamlit():
    """Launch and manage the Streamlit application process."""
    file_dir = Path(__file__).parent.absolute()
    app_path = str(file_dir / "streamlit_app.py")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port=8501"
    ]

    try:
        process = subprocess.Popen(command)
        process.wait()
    except KeyboardInterrupt:
        print("\nStreamlit server interrupted. Exiting.")
        if process.poll() is None:
             process.terminate()
             process.wait()
    except Exception as e:
        print(f"An error occurred while running Streamlit: {e}")
        sys.exit(1)
    finally:
        exit_code = process.returncode if process.poll() is not None else 1
        sys.exit(exit_code)


if __name__ == "__main__":
    run_streamlit()