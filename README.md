*This project has been created as part of the 42 curriculum by mboutte.*

# Call Me Maybe

## Description

`call-me-maybe` is a small local function-calling project built around a Hugging Face causal language model. The goal is to take a natural-language prompt, choose the most appropriate function from a predefined catalog, and extract the arguments needed to call that function.

This repository focuses on constrained decoding rather than unconstrained text generation. Instead of asking the model to freely produce a JSON blob, the implementation narrows the available next tokens during decoding so the output stays closer to the expected structure. The current pipeline:

1. validates the input function schema and prompt list with Pydantic,
2. loads a local model through the provided `llm_sdk`,
3. selects a function name with trie-constrained decoding,
4. extracts arguments one by one with type-aware constraints,
5. writes the final function calls to a JSON file.

## Instructions

### Requirements

- Python `>=3.10`
- `uv`
- enough disk space and RAM for the selected Hugging Face model

The default model is `Qwen/Qwen3-0.6B`. Other models can be passed with `--model_name`, but larger checkpoints need more resources.

Example compatible model names:

- `LiquidAI/LFM2.5-1.2B-Instruct`
- `Qwen/Qwen3.5-4B` (`+10GB` of free space recommended on the device)
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`

### Installation

```bash
make install
```

This runs:

```bash
uv sync
```

### Execution

Run the project with the default dataset:

```bash
make run
```

Equivalent command:

```bash
uv run -m src
```

Run with explicit paths:

```bash
uv run -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Run with another model:

```bash
uv run -m src --model_name Qwen/Qwen3-1.7B
```

### Linting

```bash
make lint
```

Or, for stricter typing rules:

```bash
make lint-strict
```

## Example Usage

Example prompt file entry:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```

Typical run:

```bash
uv run -m src
```

Expected output shape:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Algorithm Explanation

The core idea is to split constrained decoding into two stages.

### 1. Function selection with a token trie

At startup, every function name is tokenized with the target model tokenizer. Those token sequences are inserted into a trie-like tree. During inference, the model receives a prompt that lists the available functions and asks for `fn_name`.

Instead of allowing the whole vocabulary at each step, decoding is limited to the valid child tokens of the current trie node. The model then takes the highest-logit token among only those allowed continuations. This continues until the special stop token inserted at the end of each function path is reached.

This gives two benefits:

- the model cannot invent a function that is not present in the schema,
- decoding stays efficient because the search space is much smaller than the full vocabulary.

### 2. Argument extraction with type-aware constraints

After a function is selected, arguments are extracted sequentially. The already extracted arguments are appended back into the prompt, so each next field is generated with extra context.

The implementation currently uses two argument-generation modes:

- Constrained numeric/boolean mode:
  a restricted token set is used for integers, floats, and booleans. For numbers, only tokens corresponding to digits, minus sign, and decimal point are allowed.
- Free string mode:
  the model generates freely until a newline is produced, then the value is stripped and unquoted when possible.

This approach is simpler than full JSON-schema constrained decoding, but it demonstrates the main principle: constrain the model only where the structure matters most.

## Design Decisions

Several implementation choices shape the project:

- Pydantic validation is used before inference so malformed schemas or prompt files fail early and clearly.
- Function names are constrained with a trie instead of post-processing free text. This prevents invalid names instead of correcting them afterward.
- Argument extraction is done field by field, which keeps prompts simple and makes debugging easier.
- The local `llm_sdk` wraps Hugging Face model loading and cache-based next-token generation, which keeps the project code focused on decoding logic.
- The project favors readability and experimentation over full generality. Only a subset of parameter types is truly constrained today; arrays and objects currently fall back to free generation.

## Performance Analysis

### Accuracy

The constrained function-name stage is the strongest part of the pipeline because the model is forced to stay inside the known function catalog. Argument accuracy is more mixed:

- numeric extraction benefits from tight token constraints,
- short string extraction works reasonably well on simple prompts,
- complex phrasing or ambiguous prompts can still produce wrong values,
- free-form types are less reliable than strictly constrained types.

The sample output file shows that difficult prompts can still lead to incorrect arguments or even incorrect function choices when prompts drift away from the training examples.

### Speed

The trie-based function selection is lightweight because each step only compares a small allowed set of tokens. Cache-based decoding in `llm_sdk` also avoids recomputing the whole sequence from scratch after every generated token.

The biggest performance cost remains model loading and token-by-token inference, especially on CPU. Larger models may improve quality, but they also increase latency and memory usage.

### Reliability

The system is reliable for:

- validating the input data format,
- preventing out-of-schema function names,
- producing a consistent output JSON structure.

Its reliability is lower for complex free-form argument extraction. In practice, it works best on short, direct prompts whose wording closely matches the function descriptions.

## Challenges Faced

The main challenges in this project come from balancing control and flexibility:

- Tokenization makes constrained decoding trickier than simple string matching because function names must be constrained at the token level, not the character level.
- Stopping generation cleanly is important. Numeric decoding uses a heuristic based on newline likelihood, while string decoding stops on newline emission.
- Different argument types need different strategies. Numbers are easy to constrain; strings are much harder without a more advanced grammar-based decoder.
- Model quality strongly affects extraction quality. A lightweight model is easier to run locally, but it can be less precise on harder prompts.

The current solution addresses these issues by keeping the constrained part small and explicit: strict control for function names, partial control for arguments, and schema validation before inference.

## Testing Strategy

Validation currently relies on a combination of static checks and functional runs:

- Pydantic models verify the structure of `functions_definition.json` and the prompt dataset before inference starts.
- Sample prompts in `data/input/function_calling_tests.json` are used to exercise the pipeline end to end.
- Results are inspected in `data/output/function_calling_results.json` to compare selected functions and extracted arguments against expectations.
- Linting with `flake8` and `mypy` helps catch typing and style issues during development.

There is not yet a dedicated automated unit-test suite for the decoding logic. A natural next step would be adding repeatable tests for trie construction, function selection, stopping criteria, and argument parsing edge cases.

## Resources

Classic references related to the topic:

- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/index
- Hugging Face text generation guide: https://huggingface.co/docs/transformers/main/en/generation_strategies
- Pydantic documentation: https://docs.pydantic.dev/
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- OpenAI function calling and structured output ideas, for conceptual comparison: https://platform.openai.com/docs/guides/function-calling
- Articles and discussions on constrained decoding / structured generation from language models are also useful background for extending this project toward grammar-based or JSON-schema-based decoding.

### AI Usage

AI was used in a limited supporting role:

- to help draft and structure this `README.md`,
- to help write the current docsting of the function.
