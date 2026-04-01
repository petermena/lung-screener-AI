FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
RUN pip install --upgrade pip && pip install .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
