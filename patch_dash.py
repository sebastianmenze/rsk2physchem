import pathlib, glob, sys

matches = glob.glob("/usr/local/lib/python*/site-packages/dash/dcc/async-upload.js")
if not matches:
    sys.exit("async-upload.js not found")
p = pathlib.Path(matches[0])
src = p.read_text()

old = (
    'if(e.dataTransfer&&e.dataTransfer.items){var r=Array.from(e.dataTransfer.items),i=[];'
    'for(var o of r)if("file"===o.kind){var a=o.webkitGetAsEntry?o.webkitGetAsEntry():null;'
    'if(a){var s=yield t.traverseFileTree(a);i.push(...s)}else{var c=o.getAsFile();c&&i.push(c)}}'
    'return i}return e.target&&e.target.files?Array.from(e.target.files):e.dataTransfer&&e.dataTransfer.files?Array.from(e.dataTransfer.files):[]'
)
new = (
    'return e.dataTransfer&&e.dataTransfer.files?Array.from(e.dataTransfer.files)'
    ':e.target&&e.target.files?Array.from(e.target.files):[]'
)

out = src.replace(old, new, 1)
assert out != src, "patch target not found"
p.write_text(out)
print(f"patched {p}")
