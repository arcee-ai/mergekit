# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

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
