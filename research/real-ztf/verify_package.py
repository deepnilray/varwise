"""Check every immutable original input and code file against ZIP manifest."""

from hashlib import sha256
from pathlib import Path
import json


ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/"results"/"package_input_checksums.json").read_text())
    failures=[]
    for path,expected in manifest["sha256_by_relative_path"].items():
        file=ROOT/path
        if not file.is_file():
            failures.append("missing:"+path)
        elif sha256(file.read_bytes()).hexdigest()!=expected:
            failures.append("changed:"+path)
    print(json.dumps({"status":"PASS" if not failures else "FAIL",
                      "input_files_checked":len(manifest["sha256_by_relative_path"]),
                      "failures":failures[:30]},indent=2))
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
