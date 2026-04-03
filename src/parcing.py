
import json
from argparse import ArgumentParser

from pydantic import ValidationError

from .data_validation import FunctionList, PromptList


class ParsingError(Exception):
    """Exception raised when an error occurs during data parsing.

    Args:
        msg (str): The error message to display.
        Defaults to "An error occurred while loading data".
    """
    def __init__(self, msg: str = "An error occurred while loading data"):
        super().__init__(msg)


def input_parcing() -> tuple[FunctionList, PromptList, str, str]:
    """Parses command-line arguments and loads/validates function/promptdata.

    This function sets up an argument parser to handle input paths for
    functions definition, input data, output path, and model name. It loads
    JSON data from the specified files, validates them against Pydantic models,
    and returns the validated objects along with output and model name
    arguments.

    Args:
        None (arguments are parsed from command line).

    Returns:
        tuple[FunctionList, PromptList, str, str]: A tuple containing:
            - function_lst (FunctionList): Validated list of function
            definitions.
            - prompt_lst (PromptList): Validated list of prompts.
            - output (str): Path or format for output.
            - model_name (str): Name of the model (e.g., from Hugging Face).

    Raises:
        ParsingError: If there is an error loading or validating the JSON
            files.
    """
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="model name (on Hugging Face)"
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

    return function_lst, prompt_lst, args.output, args.model_name
