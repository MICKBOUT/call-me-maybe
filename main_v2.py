import json
from typing import Optional
import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model


class custom_llm(Small_LLM_Model):
    def __init__(
        self,
        function_file: str = "data/input/functions_definition.json",
        model_name: Optional[str] = "Qwen/Qwen3-0.6B",
        display_tree: Optional[bool] = False,
    ) -> None:
        super().__init__(
            model_name,
        )
        end_token = "<|im_end|>"
        self.end_token_id = self.encode_lst(end_token)[0]

        with open(function_file, 'r') as f:
            function_json = json.load(f)

        # Tree to choose the right function
        self.tree_function = {}
        for function in function_json:
            tokenize_function = self.encode_lst(function["name"])
            leaf = self.tree_function
            for token in tokenize_function:
                if token not in leaf:
                    leaf[token] = {"name": self.decode(token)} if display_tree else {}
                leaf = leaf[token]
            leaf[self.end_token_id] = {"name": end_token} if display_tree else {}
        # code to visulise the tree
        if display_tree:
            from tree_visualizer import build_rich_tree
            build_rich_tree(self.tree_function)

        # create a dict w/ key: value (fn_name: fn_data)
        self.function_dict = {
            function["name"]: function
            for function in function_json}

    def encode_lst(self, text: str) -> list[int]:
        return self.encode(text).tolist()[0]

    def find_fn(self, prompt):
        formated_prompt = (
           "<|im_start|>system\n"
           "You are a function selector. Given a user request, respond with ONLY the name of the most appropriate function to call. No explanation, no arguments, just the function name.<|im_end|>\n"
           "<|im_start|>user\n"
           f"{prompt}<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n\n</think>\n"
        )

        answer = ""
        tree_function = self.tree_function
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

            tree_function = tree_function[next_token]

            decoded = self.decode([next_token])
            answer += decoded

            logits, past = self.next_token_with_cache(next_token, past)

        return answer

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

            # NEXT TOKEN (fast path)
            logits, past = self.next_token_with_cache(next_token, past)
            next_token = int(np.argmax(logits))

        print(f"\n{i}/{max_new_tokens} tokens used")
        return answer


if __name__ == "__main__":

    llm = custom_llm(display_tree=False)
    # Qwen3 chat template format — hardcoded as a plain string
    # user_message = "Yes or No, am i good looking ?"
    # llm.generate(user_message, thinking=True)

    # test fn name
    path_prompt = "data/input/function_calling_tests.json"
    with open(path_prompt, 'r') as f:
        prompt_dict = json.load(f)

    result = []
    for prompt_data in prompt_dict:
        result.append(llm.find_fn(prompt_data))

    print(result)
