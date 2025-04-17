"""Emotion Analyzer Application Entry Point.

This module serves as the main entry point for the FastAPI application.
It configures and launches the ASGI server using uvicorn with appropriate settings
for production deployment.

Usage:
    python main.py

The server will start on host 0.0.0.0 (all interfaces) and port 8000 by default.
Hot reload is enabled in development mode for convenient debugging and development.
"""

import uvicorn

if __name__ == "__main__":
    # Configure and start the ASGI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Standard development port
        reload=True      # Enable hot reload for development
    )