FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch Dash 4 Upload component: when multiple=True, getDataTransferItems
# uses webkitGetAsEntry() asynchronously, but DataTransferItem objects
# become stale after the first await, silently dropping files 2..N.
# Fix: remove the branch that jumps to the webkitGetAsEntry path so all
# drops use dataTransfer.files (which stays valid across async boundaries).
RUN python3 -c "
import pathlib, re
p = pathlib.Path('/usr/local/lib/python3.11/dist-packages/dash/dcc/async-upload.js')
src = p.read_text()
# Remove 'if(t.props.multiple){n.n=3;break}' from getDataTransferItems
patched = src.replace('if(t.props.multiple){n.n=3;break}', '', 1)
assert patched != src, 'Patch target not found – check Dash version'
p.write_text(patched)
print('async-upload.js patched successfully')
"

COPY app.py .

EXPOSE 8050

CMD ["python", "app.py"]
