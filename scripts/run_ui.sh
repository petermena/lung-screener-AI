#!/usr/bin/env bash
set -euo pipefail
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
