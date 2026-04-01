import json
from typing import Optional, Any, Callable, cast
import numpy as np
import argparse

from .type_aliases import TreeNode, ConstrainedSet, TokenId
from llm_sdk import Small_LLM_Model
from .tree_visualizer import build_rich_tree


class custom_llm(Small_LLM_Model):
    def __init__(
        self,
        function_file: str = "data/input/functions_definition.json",
        model_name: str = "Qwen/Qwen3-0.6B",
        display_tree: Optional[bool] = False,
    ) -> None:
        super().__init__(
            model_name,
        )
        back_n = "\n"
        self.back_n_id = self.encode_lst(back_n)[0]
        end_token = "<|im_end|>"
        self.end_token_id = self.encode_lst(end_token)[0]

        with open(function_file, 'r') as f:
            function_json = json.load(f)

        # Tree to choose the right function
        self.tree_function: TreeNode = {
            self.end_token_id:
            {"name": end_token} if display_tree else {}
        }
        for function in function_json:
            tokenize_function = self.encode_lst(function["name"])
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

        # CONSTRAINED_SET
        self.integer_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "\n"}}

        self.number_constrained = self.integer_constrained | {
            self.encode_lst(".")[0]}

        self.boolean_constrained = {
            self.encode_lst(ell)[0] for ell in
            {"True", "False", "\n"}}

        # create a dict w/ key: value (fn_name: fn_data)
        self.function_dict = {
            function["name"]: function
            for function in function_json}

    def encode_lst(self, text: str) -> list[TokenId]:
        return cast(list[TokenId], self.encode(text).tolist()[0])

    def find_fn(self, prompt: str) -> str:
        formated_prompt = (
           "<|im_start|>system\n"
           "You are a function selector. Given a user request, "
           "respond with ONLY the name of the most appropriate "
           "function to call. No explanation, no arguments, "
           "just the function name.<|im_end|>\n"
           "<|im_start|>user\n"
           f"{prompt}<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n\n</think>\n"
        )

        answer = []
        tree_function: TreeNode = self.tree_function
        input_ids = self.encode_lst(formated_prompt)
        logits, past = self.init_generation(input_ids)
        while True:
            next_token = max(
                (k for k in tree_function if isinstance(k, int)),  # skip str
                key=lambda t: logits[t],
                default=self.end_token_id
            )

            if next_token == self.end_token_id:
                break

            tree_function = cast(TreeNode, tree_function[next_token])
            answer.append(next_token)

            logits, past = self.next_token_with_cache(next_token, past)

        return self.decode(answer)

    def find_args(self, prompt: str, fn_name: str) -> dict[str, Any]:
        def constrian_type(
          prompte: str,
          constrained_set: ConstrainedSet,
          arg_type: Callable
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
        for arg_name, arg_data in fn_data["parameters"].items():
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
                f"  {arg_name}: {arg_data['type']} ="
            )

            if arg_data['type'] == "number":
                value = constrian_type(
                    formated_prompt, self.number_constrained, float)
            elif arg_data['type'] == "integer":
                value = constrian_type(
                    formated_prompt, self.integer_constrained, int)
            elif arg_data['type'] == "boolean":
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
                    while value[0] == " ":
                        value = value[1:]
                    if value[0] == "'" and value[-1] == "'":
                        value = value[1:-1]
                    elif value[0] == '"' and value[-1] == '"':
                        value = value[1:-1]
                except Exception:
                    pass

            extracted += f"  {arg_name}: {arg_data['type']} = {value}\n"
            args[arg_name] = value

        return args

    def generate(
        self,
        prompt: str,
        thinking: bool = True,
        max_new_tokens: int = 512
    ) -> str:

        if thinking:
            formated_prompt = (
                "<|im_start|>system\n"
                "You are a assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n"
            )
        else:
            formated_prompt = (
                "<|im_start|>system\n"
                "You are a helpful assistant. /no_think<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n\n</think>\n"
            )

        input_ids = self.encode_lst(formated_prompt)

        logits, past = self.init_generation(input_ids)
        next_token = int(np.argmax(logits))
        answer = ""

        for i in range(max_new_tokens):

            if next_token == self.end_token_id:
                break

            decoded = self.decode([next_token])
            print(decoded, end="", flush=True)
            answer += decoded

            logits, past = self.next_token_with_cache(next_token, past)
            next_token = int(np.argmax(logits))

        print(f"\n{i}/{max_new_tokens} tokens used")
        return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="call_me_maybe")

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

    llm = custom_llm(
        function_file=args.functions_definition,
        display_tree=False
    )
    with open(args.input, 'r') as f:
        prompt_dict = json.load(f)

    result = []
    print()
    for prompt_data in prompt_dict:
        fn_name = llm.find_fn(prompt_data["prompt"])
        param = llm.find_args(prompt_data["prompt"], fn_name)
        print(prompt_data["prompt"])
        print(fn_name)
        print("\n".join(f"{key} = {value}" for key, value in param.items()))
        print()
        result.append({
            "prompt": prompt_data["prompt"],
            "name": fn_name,
            "parameters": param
        })

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    print("\n\n", result)


if __name__ == "__main__":
    main()
