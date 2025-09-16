#!/bin/bash

# Activate the virtual environment
source /home/emi/ai_env311/bin/activate

# Navigate to the project directory
cd /mnt/c/code/AI_Vision/multiprocessing

# Install streamlit dependencies if not already installed
pip install -r requirements_streamlit.txt

# Run the Streamlit application with WSL-friendly settings
streamlit run streamlit_detection_viewer.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
