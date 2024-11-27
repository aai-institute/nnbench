import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def parse_nnbench_config(pyproject_path):
    with open(pyproject_path, "rb") as fp:
        config = tomllib.load(fp)

    return config.get("tool", {}).get("nnbench", {})
