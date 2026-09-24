FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch Dash 4 Upload: remove the branch that routes multiple=True drops
# through webkitGetAsEntry (which goes stale after first async yield),
# so all drops use dataTransfer.files instead.
# Matched by regex because minified variable names differ between Dash
# releases (e.g. 't.props.multiple){n.n=3' vs 'this.props.multiple){e.n=3').
RUN JS=$(find / -name async-upload.js -path "*/dash/dcc/*" 2>/dev/null | head -1) && \
    echo "Patching $JS" && \
    python3 -c "import re,sys; p=__import__('pathlib').Path(sys.argv[1]); src=p.read_text(); out,n=re.subn(r'if\(\w+\.props\.multiple\)\{\w+\.n=3;break\}','',src,count=1); assert n==1,'patch target not found'; p.write_text(out); print('patched')" "$JS"

COPY app.py .

EXPOSE 8050

CMD ["python", "app.py"]
