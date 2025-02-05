# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Dict, Optional, Union

import pydantic
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.common import ModelReference


class ModelTokenEmbedding(BaseModel, frozen=True):
    kind: Literal["model_token"]
    model: ModelReference
    token_id: Optional[int] = None
    token: Optional[str] = None

    @pydantic.model_validator(mode="after")
    def validate_token(self):
        if self.token_id is None and self.token is None:
            raise ValueError("token_id or token must be specified")
        if self.token_id is not None and self.token is not None:
            raise ValueError("only one of token_id or token may be specified")
        return self


class ZeroEmbedding(BaseModel, frozen=True):
    kind: Literal["zero"]


class TokenEmbeddingConfig(BaseModel, frozen=True):
    source: Union[ModelTokenEmbedding, ZeroEmbedding, ModelReference, None] = None
    force: bool = False


class TokenizerConfig(BaseModel, frozen=True):
    source: Union[ModelReference, Literal["union"], Literal["base"]] = "union"
    tokens: Optional[Dict[str, TokenEmbeddingConfig]] = None
    pad_to_multiple_of: Optional[int] = None
