from typing import NamedTuple

from rfp import Params


class Model_Params(NamedTuple):
    body: Params
    head: Params
    bias: Params


if __name__ == "__main__":
    import rich

    print(rich.inspect(Model_Params))
