@echo off
cd /d C:\Users\Administrator\stock.administrator.vital\quant-stock-dashboard
start "" http://localhost:8501
streamlit run app.py --server.port 8501
pause
