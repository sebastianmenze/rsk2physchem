FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch Dash Upload: replace the webkitGetAsEntry branch (goes stale after
# first async yield) with a direct dataTransfer.files read for multiple drops.
COPY patch_dash.py .
RUN python3 patch_dash.py

COPY app.py .

EXPOSE 8050

CMD ["python", "app.py"]
