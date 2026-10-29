#!/bin/sh
# Modos: app (padrão) | notebook | train | qualquer outro comando
set -e

case "$1" in
  app)
    exec streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --client.toolbarMode=viewer --browser.gatherUsageStats=false
    ;;
  notebook)
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/app
    ;;
  train)
    exec python src/train.py
    ;;
  *)
    exec "$@"
    ;;
esac
