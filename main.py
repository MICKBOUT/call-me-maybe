import json
import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model


class custom_llm(Small_LLM_Model):
    def __init__(self,
                 function_path: str = "data/input/functions_definition.json"):
        super().__init__()
        # loading json
        with open(function_path, 'r') as f:
            self.fn_lst = json.load(f)

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
        request_str = '{"prompte": "' + \
            request + \
            '",\n "name": "'
        encoded_Tensor = self.encode(request_str)
        self.encoded_json = encoded_Tensor.tolist()[0]

        return self.find_next_ell(self.fn_dict, request_str)

    def find_parameter(self, request: str, function_name: str):
        parameters_lst = [
            list(ell["parameters"].keys()) for ell in self.fn_lst
            if ell["name"] == "fn_add_numbers"][0]

        request_str = '{' + \
            f'"prompte": "{request}",' + \
            f'"name": "{function_name}",' + \
            '"parameters": {'

        for parameter in parameters_lst:
            res_token = self.encode(f'"{parameter}": "').tolist()[0]
            ids = -1
            encode_request = self.encode(request_str).tolist()[0]
            while "}" not in self.decode(res_token[-1]):
                ids = -1
                lst_proba = self.get_logits_from_input_ids(
                    self.function_encoded + encode_request + res_token)
                ids = int(np.argmax(lst_proba))
                if ids == 1:  # break if the next char is " (end of json)
                    break
                res_token.append(ids)
            break
        return llm.decode(res_token)
llm = custom_llm()


# print(llm.fn_lst)

# print([list(ell["parameters"].keys()) for ell in llm.fn_lst if ell["name"] == "fn_add_numbers"][0])
# for ell in llm.fn_lst:
#     print(ell["parameters"])
path_prompt = "data/input/function_calling_tests.json"
with open(path_prompt, 'r') as f:
    prompt_dict = json.load(f)


result = []
for prompt_data in prompt_dict:
    name = llm.find_function_name(request=prompt_data["prompt"])
    param = llm.find_parameter(prompt_data["prompt"], name)
    result.append({
        "prompt": prompt_data["prompt"],
        "name": name,
        "parameters": param
    })
    print(result[-1])
    break




# output_file = "data/output/function_calling_results.json"
# with open(output_file, "w") as f:
#     json.dump(result, f, indent=4)
