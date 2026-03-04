"""
Simple HTTP Server for Frontend
Serves the HTML/CSS/JS frontend
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path


PORT = 3000
DIRECTORY = Path(__file__).parent / "frontend"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()


def main():
    print("\n" + "=" * 60)
    print(" Reddit RAG Chatbot - Frontend Server")
    print("=" * 60)
    print(f" Serving files from: {DIRECTORY}")
    print(f" Frontend URL: http://localhost:{PORT}")
    print(" API URL: http://localhost:8000/api/v1")
    print("=" * 60)
    print("\n  Make sure the API is running: python run_api.py")
    print("\nPress Ctrl+C to stop the server\n")

    # Open browser automatically
    webbrowser.open(f"http://localhost:{PORT}")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n Server stopped")


if __name__ == "__main__":
    main()
