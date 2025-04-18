"""Emotion Analyzer Application Entry Point.

This module serves as the main entry point for the Streamlit application.
It launches the Streamlit interface using subprocess.
"""

import sys
import subprocess
from pathlib import Path

def run_streamlit():
    """Run Streamlit using subprocess."""
    file_dir = Path(__file__).parent.absolute()
    app_path = str(file_dir / "streamlit_app.py")

    # Command to run Streamlit
    command = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port=8501"
    ]

    try:
        # Start the Streamlit process
        process = subprocess.Popen(command)
        # Wait for the process to complete (e.g., user stops Streamlit)
        process.wait()
    except KeyboardInterrupt:
        print("\nStreamlit server interrupted. Exiting.")
        if process.poll() is None: # Check if process is still running
             process.terminate() # Attempt to terminate the subprocess
             process.wait() # Wait for termination
    except Exception as e:
        print(f"An error occurred while running Streamlit: {e}")
        sys.exit(1)
    finally:
        # Ensure clean exit code based on whether Streamlit finished or was interrupted
        exit_code = process.returncode if process.poll() is not None else 1
        sys.exit(exit_code)


if __name__ == "__main__":
    run_streamlit()