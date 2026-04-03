from typing import Any, cast, Callable
import json

import numpy as np

from llm_sdk import Small_LLM_Model
from .data_validation import FunctionList, AllowedType
from .parcing import input_parcing, ParsingError
from .tree_visualizer import build_rich_tree
from .type_aliases import TreeNode, TokenId, ConstrainedSet


class ModelError(Exception):
    """
    Custom exception for model-related errors.
    Attributes:
        msg (str): The error message to be displayed when the exception is
            raised.
    """
    def __init__(self, msg: str = "Error While loading the model"):
        super().__init__(msg)


class custom_llm(Small_LLM_Model):
    def __init__(
        self,
        function_lst: FunctionList,
        display_tree: bool = False,
        model_name: str = "Qwen/Qwen3-0.6B",
      ) -> None:
        """
        Custom LLM model for function selection and argument extraction.

        Attributes:
            tree_function (TreeNode): A tree structure for function selection.
            function_dict (dict): A dictionary mapping function names to their
                data.
            fn_lst_str (str): A formatted string listing available functions.
            integer_constrained (set): Set of encoded integer tokens.
            number_constrained (set): Set of encoded number tokens.
            boolean_constrained (set): Set of encoded boolean tokens.

        Args:
            function_lst (FunctionList): List of functions to be managed.
            display_tree (bool): Flag to visualize the function selection tree.
            model_name (str): Name of the model to be used.
        """
        super().__init__(model_name)
        print(f"model used: {model_name}")

        back_n = "\n"
        self.back_n_id = self.encode_lst(back_n)[0]
        end_token = "<|im_end|>"
        self.end_token_id = self.encode_lst(end_token)[0]

        # Tree to choose the right function
        self.tree_function: TreeNode = {}
        #     self.end_token_id:
        #     {"name": end_token} if display_tree else {}
        # }
        for function in function_lst.functions:
            tokenize_function = self.encode_lst(function.name)
            leaf: Any = self.tree_function
            for token in tokenize_function:
                if token not in leaf:
                    leaf[token] = {"name": self.decode(token)
                                   } if display_tree else {}
                leaf = leaf[token]
            leaf[self.end_token_id] = {"name": end_token
                                       } if display_tree else {}
        # code to visulise the tree
        if display_tree:
            build_rich_tree(self.tree_function)

        # create a dict w/ key: value (fn_name: fn_data)
        self.function_dict = {
            function.name: function
            for function in function_lst.functions}

        self.fn_lst_str = "function list:"
        for function in function_lst.functions:
            self.fn_lst_str += f"name: '{function.name}'\n"
            self.fn_lst_str += f"description: '{function.description}'\n\n"

        # CONSTRAINED_SET
        self.integer_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"}}

        self.number_constrained = self.integer_constrained | {
            self.encode_lst(".")[0]}

        self.boolean_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"True", "False"}}

    def encode_lst(self, text: str) -> list[TokenId]:
        """
        Encodes a list of strings into token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            list[TokenId]: The list of token IDs.
        """
        return cast(list[TokenId], self.encode(text).tolist()[0])

    def find_fn(self, prompt: str) -> str:
        """
        Finds the appropriate function based on the prompt.

        Args:
            prompt (str): The user's prompt.

        Returns:
            str: The name of the selected function.
        """
        formated_prompt = (
            f"{self.fn_lst_str}"
            "<|im_start|>system\n"
            "Your task is to use the function call to "
            "archived the task ask in the following prompt. "
            # "(raw data / no mistake)\n"
            f"prompte = {prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"function:\n"
            "  fn_name ="
        )

        answer = []
        tree_function: TreeNode = self.tree_function
        input_ids = self.encode_lst(formated_prompt)
        logits, past = self.init_generation(input_ids)

        next_token = None
        while next_token != self.end_token_id:
            next_token = max(
                (k for k in tree_function if isinstance(k, int)),  # skip str
                key=lambda t: logits[t],
                default=self.end_token_id
            )

            tree_function = cast(TreeNode, tree_function[next_token])
            answer.append(next_token)

            logits, past = self.next_token_with_cache(next_token, past)

        return self.decode(answer)

    def find_args(self, prompt: str, fn_name: str) -> dict[str, Any]:
        """
        Finds the arguments for a given function based on the prompt.

        Args:
            prompt (str): The user's prompt.
            fn_name (str): The name of the function.

        Returns:
            dict[str, Any]: The arguments for the function.
        """
        def generate_arg_constrian(
            prompte: str,
            constrained_set: ConstrainedSet,
            arg_type: Callable[[str], Any]
          ) -> Any:
            """
            Generates an argument based on the prompt and constrained set.

            Args:
                prompte (str): The prompt for argument generation.
                constrained_set (ConstrainedSet): The set of allowed tokens.
                arg_type (Callable[[str], Any]): The type to cast the generated
                    argument to.

            Returns:
                Any: The generated argument.
            """
            arg: list[TokenId] = []
            input_ids = self.encode_lst(prompte)
            logits, past = self.init_generation(input_ids)
            while True:
                next_token = max(
                    constrained_set,
                    key=lambda t: logits[t],
                )

                if arg and logits[self.back_n_id] > logits[next_token]:
                    break

                arg.append(next_token)
                logits, past = self.next_token_with_cache(next_token, past)
            return arg_type(self.decode(arg))

        def generate_arg_free(prompt: str) -> str:
            """
            Generate a string from a given prompt without any arguments.

            This function encodes the provided prompt, initializes the
            generation process, and iteratively generates tokens until a
            newline character is encountered. It processes the generated
            tokens to construct a final output string, ensuring that any
            leading or trailing quotes are removed.

            Args:
                prompt (str): The input prompt to generate text from.

            Returns:
                str: The generated text, trimmed of leading/trailing
                    whitespace and quotes if applicable.
            """
            input_ids = self.encode_lst(formated_prompt)
            logits, past = self.init_generation(input_ids)
            next_token = int(np.argmax(logits))
            value = ""

            next_token_value = self.decode(next_token)
            while "\n" not in next_token_value:
                value += next_token_value

                logits, past = self.next_token_with_cache(next_token, past)
                next_token = int(np.argmax(logits))
                next_token_value = self.decode(next_token)

            for c in next_token_value:
                if c == "\n":
                    break
                value += c
            value = value.lstrip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {
              "'", '"'}:
                value = value[1:-1]
            return value

        CONSTRAINED = {
            AllowedType.FLOAT: (self.number_constrained, float),
            AllowedType.INT:   (self.integer_constrained, int),
            AllowedType.BOOL:  (self.boolean_constrained, bool),
        }

        fn_data = self.function_dict[fn_name]
        args, extracted = {}, ""

        for arg_name, arg_data in fn_data.parameters.items():
            formated_prompt = (
                "<|im_start|>system\n"
                "Your task is to use the function call to "
                "archived the task ask in the following prompt. "
                "(raw data / no mistake)\n"
                f"prompte = {prompt}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"function:\n"
                f"  fn_name = {fn_name}\n"
                # f"  Description = {fn_data["description"]}\n"
                "args:\n"
                f"{extracted}"
                f"  {arg_name}: {arg_data.type.value} ="
            )

            if arg_data.type in CONSTRAINED:
                cset, cast = CONSTRAINED[arg_data.type]
                value = generate_arg_constrian(formated_prompt, cset, cast)
            else:
                value = generate_arg_free(formated_prompt)

            extracted += f"  {arg_name}: {arg_data.type.value} = {value}\n"
            args[arg_name] = value
            print(f"{arg_name} = {value}")

        return args


def main() -> None:
    """
    Main entry point for CLI.
    Parses input, init LLM, extracts function calls from prompts,
    and writes results to output JSON.

    Returns:
        None
    """
    try:
        function_lst, prompt_lst, output_file, model_name = input_parcing()
    except ParsingError:
        return

    try:
        llm = custom_llm(
            function_lst=function_lst,
            display_tree=False,
            model_name=model_name
        )
    except Exception as e:
        print(e)
        return

    result = []
    print()
    for prompt_data in prompt_lst.prompts:
        print(f"prompt: '{prompt_data.prompt}'")
        fn_name = llm.find_fn(prompt_data.prompt)
        print(f"fn_used: '{fn_name}'")
        param = llm.find_args(prompt_data.prompt, fn_name)
        print()
        result.append({
            "prompt": prompt_data.prompt,
            "name": fn_name,
            "parameters": param
        })

    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
    except Exception:
        print("Error will writing the output")


if __name__ == "__main__":
    main()
