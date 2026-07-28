#!/usr/bin/env python3
"""

// Claude generated: only for personal use by ravi.k.ayyala@fau.de
Augments compile_commands.json with entries for hand-written .cu files that
are only compiled indirectly through a one-line .hip wrapper
(`#include "foo.cu"`), so clangd/cpptools can resolve symbols inside them.

Usage: python3 generate_ide_compile_commands.py <build_dir> [output_dir]
"""
import json
import sys
from pathlib import Path

def main():
    build_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / ".clangd-db"
    out_dir.mkdir(exist_ok=True)

    db = json.loads((build_dir / "compile_commands.json").read_text())

    extra = []
    for entry in db:
        file_path = Path(entry["file"])
        if file_path.suffix != ".hip":
            continue
        cu_path = file_path.with_suffix(".cu")
        if not cu_path.exists():
            continue
        clone = dict(entry)
        clone["file"] = str(cu_path)
        if "command" in clone:
            clone["command"] = clone["command"].replace(str(file_path), str(cu_path))
        if "arguments" in clone:
            clone["arguments"] = [str(cu_path) if a == str(file_path) else a for a in clone["arguments"]]
        if "output" in clone:
            clone.pop("output")
        extra.append(clone)

    merged = db + extra
    (out_dir / "compile_commands.json").write_text(json.dumps(merged, indent=1))
    print(f"Wrote {len(merged)} entries ({len(extra)} synthetic .cu entries) to {out_dir / 'compile_commands.json'}")

if __name__ == "__main__":
    main()
