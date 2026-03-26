import json
import numpy as np

from typing import Optional

from llm_sdk.llm_sdk import Small_LLM_Model
from tree_visualizer import build_rich_tree_fn


class custom_llm(Small_LLM_Model):
    def __init__(
        self,
        function_path: Optional[str] = "data/input/functions_definition.json",
        model_name: Optional[str] = None
      ):
        super().__init__(model_name) if model_name else super().__init__()
        self.tokenize_back_n = self.encode_to_list("\n")

        # loading json
        with open(function_path, 'r') as f:
            fn_lst_dict = json.load(f)

        # dict w/ key: value (name: json_data)
        self.dict_function = {}
        for d in fn_lst_dict:
            self.dict_function[d["name"]] = d

        self.fn_dict = {}
        for function in fn_lst_dict:
            fn_tokenized = self.encode_to_list(function["name"])
            leaf = self.fn_dict
            for token in fn_tokenized:
                if token not in leaf:
                    leaf[token] = {}
                leaf = leaf[token]
            leaf[1] = None  # set an end to the tree (avoid infinit loop)
        build_rich_tree_fn(self.fn_dict)

        # json to str
        function_str = str(fn_lst_dict) + '\n'

        # function encoded to ids
        self.function_encoded = self.encode_to_list(function_str)

    def encode_to_list(self, text: str) -> list[int]:
        return self.encode(text).tolist()[0]

    def find_next_ell(self, tree_posibility: dict, request: str) -> str:
        res_token = []
        leaf = tree_posibility
        proba, ids = float("-inf"), -1

        encode_request = self.encode_to_list(request)
        while leaf:
            proba, ids = float("-inf"), -1
            lst_proba = self.get_logits_from_input_ids(
                self.function_encoded + encode_request + res_token)
            for ell in leaf.keys():
                if lst_proba[ell] > proba:
                    proba, ids = lst_proba[ell], ell
            if ids == 1:  # break if the next char is " (end of json)
                break
            res_token.append(ids)
            leaf = leaf[ids]
        return self.decode(res_token)

    def find_function_name(self, request: str):
        request_str = f"Prompt = {request}\n  fn_used ="
        self.encoded_json = self.encode_to_list(request_str)

        return self.find_next_ell(self.fn_dict, request_str)

    def readable_function(self, function_name: str) -> str:
        fn_used = self.dict_function[function_name]
        res = "Fn:\n"
        res += f'  Name: {fn_used["name"]}\n'
        res += f'  Description: {fn_used["description"]}\n'
        res += '  Parameters info:\n'
        for key, val in fn_used["parameters"].items():
            res += f"    {key}: {val['type']}\n"
        res += "\n"
        return res

    def find_parameter(self, request: str, function_name: str):
        parameters_lst = list(self.dict_function[function_name]["parameters"].keys())

        encoded_fn_info = self.encode_to_list(
            self.readable_function(function_name))
        request_str = f"Prompt = {request}\n" +\
            "  Parameters:\n"
        encode_request = self.encode_to_list(request_str)

        return_dict = {}
        encoded_arg_name = []

        static_prefix = encoded_fn_info + encode_request
        for parameter in parameters_lst:
            res_token = []
            encoded_arg_name += self.encode_to_list(f"    {parameter} = ")
            while True:
                lst_proba = np.array(self.get_logits_from_input_ids(
                    static_prefix + encoded_arg_name + res_token))
                ids = int(np.argmax(lst_proba))
                ids_decodes = self.decode(ids)
                if "\n" in ids_decodes:
                    if ids_decodes[0] != "\n":
                        to_add = ""
                        for car in ids_decodes:
                            if car == '\n':
                                break
                            to_add += car
                        res_token += self.encode_to_list(to_add)
                    break
                res_token.append(ids)
            if parameter != parameters_lst[-1]:
                encoded_arg_name += res_token + self.tokenize_back_n
            return_dict[parameter] = self.decode(res_token)
        for key, val in return_dict.items():
            val = val.strip()
            try:
                val = int(val)
            except Exception:
                if ("\"" in val[0] or '\'' in val[0]) and (
                   "\"" in val[-1] or '\'' in val[-1]):
                    val = val[1:-1]
            return_dict[key] = val
        return return_dict


def main() -> None:
    llm = custom_llm("data/input/functions_definition.json")

    path_prompt = "data/input/function_calling_tests.json"
    with open(path_prompt, 'r') as f:
        prompt_dict = json.load(f)

    result = []
    for prompt_data in prompt_dict:
        fn_name = llm.find_function_name(request=prompt_data["prompt"])
        param = llm.find_parameter(prompt_data["prompt"], fn_name)
        print(param)
        result.append({
            "prompt": prompt_data["prompt"],
            "name": fn_name,
            "parameters": param
        })

    output_file = "data/output/function_calling_results.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)



if __name__ == "__main__":
    main()
