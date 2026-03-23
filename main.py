import json
import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model
# from tree_visualizer import build_rich_tree_fn


class custom_llm(Small_LLM_Model):
    def __init__(self,
                 function_path: str = "data/input/functions_definition.json"):
        super().__init__()
        # loading json
        with open(function_path, 'r') as f:
            self.fn_lst = json.load(f)

        self.dict_function = {}
        for d in self.fn_lst:
            self.dict_function[d["name"]] = d
        # build_rich_tree_fn(self.dict_function)

        self.fn_dict = {}
        for fn in self.fn_lst:
            tokens = self.encode(fn["name"]).tolist()[0]
            leaf = self.fn_dict
            for token in tokens:
                if token not in leaf:
                    leaf[token] = {}
                leaf = leaf[token]
            leaf[1] = None  # set an end to the tree (avoid infinit loop)

        # json to str
        self.function_str = str(self.fn_lst) + '\n'

        # function encoded to ids
        encoded_Tensor = self.encode(self.function_str)
        self.function_encoded = encoded_Tensor.tolist()[0]

    def find_next_ell(self, tree_posibility: dict, request: str) -> str:
        res_token = []
        leaf = tree_posibility
        proba, ids = float("-inf"), -1

        encode_request = self.encode(request).tolist()[0]
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
        request_str = '{"prompt": "' + \
            request + \
            '", "name": "'
        encoded_Tensor = self.encode(request_str)
        self.encoded_json = encoded_Tensor.tolist()[0]

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
        parameters_lst = [
            list(ell["parameters"].keys()) for ell in self.fn_lst
            if ell["name"] == function_name][0]

        encoded_fn_info = self.encode(
            self.readable_function(function_name)).tolist()[0]
        request_str = f"Prompt = {request}\n" +\
            "  Parameters:\n"
        encode_request = self.encode(request_str).tolist()[0]

        return_dict = {}
        encoded_arg_name = []
        for parameter in parameters_lst:
            res_token = []
            encoded_arg_name += self.encode(f"    {parameter} = ").tolist()[0]
            while True:
                # print("========== \n Request:\n", self.decode(
                #     encoded_fn_info + encode_request +
                #     encoded_arg_name + res_token))
                lst_proba = self.get_logits_from_input_ids(
                    encoded_fn_info + encode_request +
                    encoded_arg_name + res_token)
                ids = int(np.argmax(lst_proba))
                # print("res:\n", self.decode(ids))
                ids_decodes = self.decode(ids)
                if "\n" in ids_decodes:
                    if ids_decodes[0] != "\n":
                        to_add = ""
                        for car in ids_decodes:
                            if car == '\n':
                                break
                            to_add += car
                        res_token += self.encode(to_add).tolist()[0]
                    break
                res_token.append(ids)
            if parameter != parameters_lst[-1]:
                encoded_arg_name += res_token + self.encode("\n").tolist()[0]
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


llm = custom_llm()

path_prompt = "data/input/function_calling_tests.json"
with open(path_prompt, 'r') as f:
    prompt_dict = json.load(f)


result = []
for prompt_data in prompt_dict:
    name = llm.find_function_name(request=prompt_data["prompt"])
    param = llm.find_parameter(prompt_data["prompt"], name)
    print(param)
    result.append({
        "prompt": prompt_data["prompt"],
        "name": name,
        "parameters": param
    })

print(result)

output_file = "data/output/function_calling_results.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=4)
