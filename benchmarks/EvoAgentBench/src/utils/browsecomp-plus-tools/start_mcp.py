#!/usr/bin/env python3
"""Start the MCP search server for BrowseComp-Plus.

Reads mcp_server section from information_retrieval.yaml and launches the
FAISS search server.

Usage:
    python start_mcp.py                          # default config
    python start_mcp.py /path/to/custom.yaml     # custom config
"""

import os
import sys

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCHER_DIR = os.path.join(_DIR, "searcher")


def main():
    # Auto-detect JDK from conda env (pyserini/jnius needs JAVA_HOME + JVM_PATH)
    prefix = sys.prefix
    if not os.environ.get("JAVA_HOME"):
        if os.path.exists(os.path.join(prefix, "bin", "java")):
            os.environ["JAVA_HOME"] = prefix
    if not os.environ.get("JVM_PATH"):
        jvm = os.path.join(prefix, "lib", "jvm", "lib", "server", "libjvm.so")
        if os.path.exists(jvm):
            os.environ["JVM_PATH"] = jvm

    _default_yaml = os.path.join(_DIR, "..", "..", "domains", "information_retrieval", "information_retrieval.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else _default_yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["mcp_server"]

    # Resolve relative paths against yaml file directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("index_path", "model_name"):
        val = cfg.get(key, "")
        if not val or os.path.isabs(val):
            continue
        resolved = os.path.join(config_dir, val)
        # Only resolve if it looks like a local path (has .. or the resolved path exists)
        if val.startswith(".") or os.path.exists(resolved.split("*")[0].rstrip("/")):
            cfg[key] = os.path.abspath(resolved)

    # Make BrowseComp-Plus searcher importable
    if _SEARCHER_DIR not in sys.path:
        sys.path.insert(0, _SEARCHER_DIR)

    argv = [
        "mcp_server",
        "--searcher-type", cfg.get("searcher_type", "faiss"),
        "--index-path", cfg["index_path"],
        "--transport", "sse",
        "--port", str(cfg.get("port", 9100)),
        "--k", str(cfg.get("k", 5)),
        "--snippet-max-tokens", str(cfg.get("snippet_max_tokens", 512)),
    ]
    if cfg.get("model_name"):
        argv += ["--model-name", cfg["model_name"]]
    sys.argv = argv

    from mcp_server import main as mcp_main
    mcp_main()


if __name__ == "__main__":
    main()
