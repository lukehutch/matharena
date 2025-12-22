FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    bash \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy scipy sympy pandas

RUN useradd -m sandboxuser
USER sandboxuser
CMD ["python3"]