from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict


class AllowedType(str, Enum):
    INT = "integer"
    FLOAT = "number"
    STR = "string"
    BOOL = "boolean"
    LIST = "array"
    DICT = "object"


class ParameterType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: AllowedType


class FunctionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[Annotated[str, Field(min_length=1)], ParameterType]
    returns: ParameterType


class FunctionList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    functions: list[FunctionItem]


class PromptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class PromptList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompts: list[PromptItem]
