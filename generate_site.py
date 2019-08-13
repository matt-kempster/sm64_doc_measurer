#!/usr/bin/env python3.7

import argparse
import http.server
import os
import shutil
import socketserver
from pathlib import Path

from index import render

# The following terrible wildcard import is necessitated by my local pickle cache.
# See https://stackoverflow.com/q/27732354/11799075. I'll have to go back and
# fix this later.
from sm64_parse import *

PORT = 8000


def serve_site(site_output: Path, port: int) -> None:
    os.chdir(site_output)
    with socketserver.TCPServer(
        ("", port), http.server.SimpleHTTPRequestHandler
    ) as httpd:
        print("serving at port", port)
        httpd.serve_forever()


def main_site() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="PATH_TO_SM64_SOURCE_DIR")
    args = parser.parse_args()

    root = Path(args.root)
    results = sm64_parse(root, commits_to_analyze=50)

    parent = Path(__file__).parent
    site_output = parent / "_site"

    site_output.mkdir(exist_ok=True)
    (site_output / "index.html").write_text(render({"results": results}))
    if not (site_output / "js").is_dir():
        shutil.copytree(parent / "js", site_output / "js")
    if not (site_output / "graph.css").is_file():
        shutil.copy(parent / "graph.css", site_output / "graph.css")

    serve_site(site_output, PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main_site())
