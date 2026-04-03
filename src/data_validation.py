from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict


class AllowedType(str, Enum):
    """
    An enumeration of allowed types for parameters.
    """
    INT = "integer"
    FLOAT = "number"
    STR = "string"
    BOOL = "boolean"
    LIST = "array"
    DICT = "object"


class ParameterType(BaseModel):
    """
    A Pydantic model for validating a parameter type.

    Attributes:
        type (AllowedType): The type of the parameter.
    """
    model_config = ConfigDict(extra="forbid")
    type: AllowedType


class FunctionItem(BaseModel):
    """
    A Pydantic model for validating a single function item.

    Attributes:
        name (str): The name of the function.
        description (str): A description of the function.
    """
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[Annotated[str, Field(min_length=1)], ParameterType]
    returns: ParameterType


class FunctionList(BaseModel):
    """
    A Pydantic model for validating a list of function items.

    Attributes:
        functions (list[FunctionItem]): A list containing FunctionItem
            instances.
    """
    model_config = ConfigDict(extra="forbid")
    functions: list[FunctionItem]


class PromptItem(BaseModel):
    """
    A Pydantic model for validating a single prompt item.

    Attributes:
        prompt (str): The prompt string.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class PromptList(BaseModel):
    """
    A Pydantic model for validating a list of prompt items.

    Attributes:
        prompts (list[PromptItem]): A list containing PromptItem instances.
    """
    model_config = ConfigDict(extra="forbid")
    prompts: list[PromptItem]
