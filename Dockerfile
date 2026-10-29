FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m ipykernel install --name fiap-tech-challenge-fase01 \
       --display-name "Python (fiap-tech-challenge-fase01)"

COPY database/ database/
COPY src/ src/
COPY app/ app/
COPY tech-challenge.ipynb README.md ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# O modelo é treinado durante o build: a imagem já sai com ele pronto para uso.
RUN python src/train.py

EXPOSE 8501 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["app"]
