def main() -> None:
    try:
        raise TypeError("bad type")
    except TypeError as type_error:
        type_error.add_note("What up beck")
        raise


if __name__ == "__main__":
    main()
# from copy import deepcopy
# from typing import Any, TypeVar

# T = TypeVar("T")

# def copyof(value: T) -> T:
#     return deepcopy(value)

# name : str = "Anthony"
# copy_of_name = copyof(name)
# print(name)
# print(copy_of_name)
