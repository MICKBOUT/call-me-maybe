
import json
from argparse import ArgumentParser

from pydantic import ValidationError

from .data_validation import FunctionList, PromptList


class ParsingError(Exception):
    def __init__(self, msg: str = "An error occurred while loading data"):
        super().__init__(msg)


def input_parcing() -> tuple[FunctionList, PromptList, str]:
    parser = ArgumentParser(description="call_me_maybe")

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path or content of the functions definition"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Input data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Output path or format"
    )
    args = parser.parse_args()

    try:
        with open(args.functions_definition, 'r') as f:
            function_lst = json.load(f)
    except Exception:
        print("Error will loading the function file")
        raise ParsingError()
    try:
        function_lst = FunctionList.model_validate({"functions": function_lst})
    except ValidationError as e:
        print("Error while loading function lst")
        for error in e.errors():
            locs = "/".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            print(f"Error at \"{locs}\": {msg}")
        raise ParsingError()

    try:
        with open(args.input, 'r') as f:
            prompt_lst = json.load(f)
    except Exception:
        print("Error will loading the prompte")
        raise ParsingError
    try:
        prompt_lst = PromptList.model_validate({"prompts": prompt_lst})
    except ValidationError as e:
        print("Error while loading prompt lst")
        for error in e.errors():
            locs = "/".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            print(f"Error at \"{locs}\": {msg}")
        raise ParsingError()

    return function_lst, prompt_lst, args.output
