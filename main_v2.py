import json
from typing import Optional, Any
import numpy

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

        self.end_token_id = self.encode_lst("<|im_end|>")[0]

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

    def generate(
            self,
            prompt: str,
            thinking: bool = True,
            max_new_tokens: int = 512
      ) -> str:
        if thinking:
            formated_prompte = (
                "<|im_start|>system\n"
                "You are a assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n"
            )
        else:
            formated_prompte = (
                "<|im_start|>system\n"
                "You are a helpful assistant. /no_think<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n\n"
                "</think>\n"
            )

        # encode the formated_prompte
        input_ids = self.encode_lst(formated_prompte)
        generated = []
        answer = ""
        for i in range(max_new_tokens):
            logits = self.get_logits_from_input_ids(input_ids + generated)

            # greedy: pick highest logit
            next_token = int(numpy.argmax(logits))

            # stop if end token
            if next_token == self.end_token_id:
                break

            generated.append(next_token)
            decoded_token = self.decode(next_token)
            print(self.decode(next_token), end="", flush=True)
            answer += decoded_token
        print(f"{i}/{max_new_tokens} token used")
        return answer


if __name__ == "__main__":
    llm = custom_llm(display_tree=False)

    # Qwen3 chat template format — hardcoded as a plain string
    user_message = "what is the height of an average beluga in cm ?"

    llm.generate(user_message, thinking=True)
