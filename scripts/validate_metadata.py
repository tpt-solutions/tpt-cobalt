"""Validate first-party crate metadata: README/CHANGELOG presence, keywords
(<= 5, lowercase), valid crates.io categories, and required manifest fields.

Used by CI (.github/workflows/ci.yml). Exit 1 on any problem."""

import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # py < 3.11
    tomllib = None

VALID_CATEGORIES = {
    "accessibility", "api-bindings", "asynchronous", "authentication",
    "caching", "command-line-utilities", "compilers", "compression", "config",
    "concurrency", "command-line-interface", "database-implementations",
    "data-structures", "date-and-time", "development-tools", "development-tools::build-utils",
    "development-tools::debugging", "development-tools::ffi", "development-tools::formatting",
    "development-tools::parsing", "development-tools::parsing::implementations",
    "development-tools::profiling", "development-tools::testing", "development-tools",
    "embedded", "emulators", "encoding", "filesystem", "game-development", "games",
    "graphics", "gui", "hardware-support", "http-client", "http-server", "images",
    "internationalization", "multimedia::audio", "multimedia::images",
    "multimedia::video", "multimedia", "memory-management", "network-programming",
    "no-std", "os", "parser-implementations", "parser-tools", "rendering::data-formats",
    "rendering::engine", "rendering::graphics-api", "rendering", "rust-patterns",
    "science", "science::robotics", "algorithms", "template-engine", "text-editors",
    "text-processing", "value-formatting", "visualization", "wasm", "web-programming",
    "web-programming::http-client", "web-programming::http-server",
    "web-programming::websocket", "windows-api",
}


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..", "crates")
    problems = []
    n = 0
    for d in sorted(os.listdir(root)):
        manifest = os.path.join(root, d, "Cargo.toml")
        if not os.path.isfile(manifest):
            continue
        n += 1
        if tomllib is None:
            text = open(manifest, encoding="utf-8").read()
            meta = {}
            for field in ("description", "readme", "license"):
                if f"{field} =" not in text:
                    problems.append(f"{d}: missing {field}")
            mkw = re.search(r"keywords = \[(.*?)\]", text, re.S)
            mcat = re.search(r"categories = \[(.*?)\]", text, re.S)
            keywords = re.findall(r'"([^"]+)"', mkw.group(1)) if mkw else []
            categories = re.findall(r'"([^"]+)"', mcat.group(1)) if mcat else []
        else:
            meta = tomllib.load(open(manifest, "rb"))["package"]
            for field in ("description", "readme", "keywords", "categories", "license"):
                if field not in meta:
                    problems.append(f"{d}: missing {field}")
            keywords = meta.get("keywords", [])
            categories = meta.get("categories", [])
        if len(keywords) > 5:
            problems.append(f"{d}: more than 5 keywords")
        for k in keywords:
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", k):
                problems.append(f"{d}: bad keyword '{k}'")
        for c in categories:
            if c not in VALID_CATEGORIES:
                problems.append(f"{d}: unknown category '{c}'")
        for doc in ("README.md", "CHANGELOG.md"):
            if not os.path.isfile(os.path.join(root, d, doc)):
                problems.append(f"{d}: missing {doc}")
        readme = os.path.join(root, d, "README.md")
        if os.path.isfile(readme) and open(readme, encoding="utf-8").read().count("\n") < 30:
            problems.append(f"{d}: README too short (<30 lines)")
    print(f"validated {n} crates")
    for p in problems:
        print(" -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
