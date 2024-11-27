import os
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# TODO: It's not just Any in the return, but a few well-defined context keys -
# put some effort into parsing them
def parse_nnbench_config(pyproject_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(pyproject_path, "rb") as fp:
        config = tomllib.load(fp)

    return config.get("tool", {}).get("nnbench", {})
