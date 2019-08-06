import argparse
import http.server
import os
import socketserver
from pathlib import Path

from index import render
# The following terrible wildcard import is necessitated by my local pickle cache.
# See https://stackoverflow.com/q/27732354/11799075. I'll have to go back and
# fix this later.
from sm64_parse import *

PORT = 8000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="PATH_TO_SM64_SOURCE_DIR")
    args = parser.parse_args()
    root = Path(args.root)
    results = sm64_parse(root, commits_to_analyze=50)

    site_output = Path(__file__).parent / "_site"
    site_output.mkdir(exist_ok=True)

    (site_output / "index.html").write_text(render({"results": results}))

    Handler = http.server.SimpleHTTPRequestHandler

    os.chdir(site_output)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()
