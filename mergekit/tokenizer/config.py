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

from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.common import ModelReference


class TokenAverageEmbedding(BaseModel, frozen=True):
    tag: Literal["token_average"]
    model: ModelReference
    text: Optional[str] = None


class ZeroEmbedding(BaseModel, frozen=True):
    tag: Literal["zero"]


class TokenEmbeddingConfig(BaseModel, frozen=True):
    source: Union[ModelReference, TokenAverageEmbedding, ZeroEmbedding, None] = None
    force: bool = False


class TokenizerConfig(BaseModel, frozen=True):
    source: Union[ModelReference, Literal["union"], Literal["base"]] = "union"
    tokens: Optional[Dict[str, TokenEmbeddingConfig]] = None
