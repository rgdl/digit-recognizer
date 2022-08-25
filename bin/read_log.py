import json
import sys

if __name__ == "__main__":
    assert (
        len(sys.argv) > 1
    ), "Provide a kaggle log, i.e. output of `kaggle kernels output [kernel]`"

    with open(sys.argv[1], "r") as f:
        log = json.load(f)

    for item in log:
        print(item["data"], file=getattr(sys, item["stream_name"]))
