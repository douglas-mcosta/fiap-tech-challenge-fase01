FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# O modelo já vem treinado (src/train.py); o dataset de 2 GB não entra na imagem.
COPY models/ models/
COPY exemplos/ exemplos/
COPY src/ src/
COPY app/ app/
COPY Tech_Challenge_Fase1_Visao_Computacional.ipynb ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["app"]
