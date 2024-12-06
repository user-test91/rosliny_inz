#!/bin/sh

# Start FastAPI in the background
uvicorn app:app --host 127.0.0.1 --port 8000 &

# Start Streamlit in the foreground
streamlit run streamlit_app.py --server.port=8501 --server.address=127.0.0.1
