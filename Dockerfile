FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY smoobu_pipeline.py .

# Utilisateur sans droits root. /data garde l'état entre deux cycles et deux redémarrages
# (pipeline_blocked.json) : c'est un volume Docker.
RUN useradd --create-home pipeline && mkdir /data && chown pipeline /data
USER pipeline
WORKDIR /data
VOLUME /data

# Toutes les 45 min : e-mails -> extraction -> backend. La revue et les tests sont faits par le CI.
ENTRYPOINT ["python", "/app/smoobu_pipeline.py"]
CMD ["--skip-checks", "--every", "45"]
