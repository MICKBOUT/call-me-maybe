from typing import Any, Callable, cast
import json

import numpy as np

from llm_sdk import Small_LLM_Model
from .data_validation import FunctionList, AllowedType
from .parcing import input_parcing, ParsingError
from .tree_visualizer import build_rich_tree
from .type_aliases import TreeNode, ConstrainedSet, TokenId


class ModelNotFoundError(Exception):
    def __init__(self, msg: str = "Model Not found"):
        super().__init__(msg)


class custom_llm(Small_LLM_Model):
    def __init__(
        self,
        function_lst: FunctionList,
        display_tree: bool = False,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        try:
            super().__init__(model_name)
        except OSError:
            raise ModelNotFoundError()

        back_n = "\n"
        self.back_n_id = self.encode_lst(back_n)[0]
        end_token = "<|im_end|>"
        self.end_token_id = self.encode_lst(end_token)[0]

        # Tree to choose the right function
        self.tree_function: TreeNode = {
            self.end_token_id:
            {"name": end_token} if display_tree else {}
        }
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

        self.functions_description = "function list:"
        for function in function_lst.functions:
            self.functions_description += f"name: '{function.name}'\n"
            self.functions_description += f"description: '{function.description}'\n\n"

        # CONSTRAINED_SET
        self.integer_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "\n"}}

        self.number_constrained = self.integer_constrained | {
            self.encode_lst(".")[0]}

        self.boolean_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"True", "False", "\n"}}

    def encode_lst(self, text: str) -> list[TokenId]:
        return cast(list[TokenId], self.encode(text).tolist()[0])

    def find_fn(self, prompt: str) -> str:
        formated_prompt = (
            f"{self.functions_description}"
            "<|im_start|>system\n"
            "Your task is to use the function call to "
            "archived the task ask in the following prompt. "
            "(raw data / no mistake)\n"
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
        def constrian_type(
          prompte: str,
          constrained_set: ConstrainedSet,
          arg_type: Callable[[str], Any]
        ) -> Any:
            arg = []
            input_ids = self.encode_lst(prompte)
            logits, past = self.init_generation(input_ids)
            while True:
                next_token = max(
                    (k for k in constrained_set),
                    key=lambda t: logits[t],
                    default=self.end_token_id
                )

                if next_token == self.back_n_id:
                    break
                arg.append(next_token)

                logits, past = self.next_token_with_cache(next_token, past)
            value = arg_type(self.decode(arg))
            return value

        fn_data = self.function_dict[fn_name]
        args = {}

        extracted = ""
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

            if arg_data.type == AllowedType.FLOAT:
                value = constrian_type(
                    formated_prompt, self.number_constrained, float)
            elif arg_data.type == AllowedType.INT:
                value = constrian_type(
                    formated_prompt, self.integer_constrained, int)
            elif arg_data.type == AllowedType.BOOL:
                value = constrian_type(
                    formated_prompt, self.boolean_constrained, bool)
            else:
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
                try:
                    value = value.lstrip()
                    if value[0] == "'" == value[-1] == "'":
                        value = value[1:-1]
                    elif value[0] == '"' == value[-1] == '"':
                        value = value[1:-1]
                except Exception:
                    pass

            extracted += f"  {arg_name}: {arg_data.type.value} = {value}\n"
            args[arg_name] = value
            print(f"{arg_name} = {value}")

        return args


def main() -> None:
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
    except ModelNotFoundError as e:
        print(f"Error: Model '{model_name}' not found")
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
