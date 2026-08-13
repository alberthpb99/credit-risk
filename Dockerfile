# 1. Versión de Python
FROM python:3.12.1-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Agregamos archivos al directorio de trabajo
COPY requirements.txt app.py ./
COPY models/model_pipeline.joblib ./models/
COPY data/processed.csv ./data/

# 4. Instalamos las librerías
RUN pip install --no-cache-dir -r requirements.txt

# 5. Puerto de Streamlit
EXPOSE 8501

# 6. Ejecutamos
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]