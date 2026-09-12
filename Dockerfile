FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG REPO_DIR=/workspace/sahc-lightning
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Rome
ENV REPO_DIR=${REPO_DIR}
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=${REPO_DIR}/.venv
ENV QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

COPY entrypoint.sh /opt/app/entrypoint.sh
COPY --chmod=755 run_script.sh /usr/local/bin/run_script.sh

WORKDIR ${REPO_DIR}

ENTRYPOINT ["bash", "/opt/app/entrypoint.sh"]
CMD ["bash"]
