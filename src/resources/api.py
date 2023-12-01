# /src/api.py

import errno
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/data")
def read_data():
    return {"Hello": "World"}

# Run server with uvicorn (and check for port collisions)
def run_server(port=8000):
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"Port {port} is in use. Server already running.")
        else:
            raise

if __name__ == "__main__":
    run_server()