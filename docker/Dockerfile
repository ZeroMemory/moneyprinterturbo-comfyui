FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /MoneyPrinterTurbo
RUN git clone --depth 1 --branch v1.3.9 https://github.com/harry0703/MoneyPrinterTurbo.git . \
    && test "$(git rev-parse HEAD)" = "6992380c5e82f07f3e1fc46b90927b6f1c655407" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY install.py configure_ltx.py supported-releases.json LICENSE /opt/mpt-comfyui/
COPY payload/ /opt/mpt-comfyui/payload/
COPY mpt/ /opt/mpt-comfyui/mpt/
COPY runtime/ /opt/mpt-comfyui/runtime/
COPY docker/entrypoint.py /opt/mpt-comfyui/docker/entrypoint.py
RUN python -B /opt/mpt-comfyui/install.py install /MoneyPrinterTurbo

ENV MPT_HOME=/data \
    HOME=/data \
    XDG_CACHE_HOME=/data/.cache \
    PYTHONPATH=/MoneyPrinterTurbo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-B", "/opt/mpt-comfyui/docker/entrypoint.py"]
