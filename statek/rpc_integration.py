# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#pylint: disable=R0903
import functools
from typing import Any


class _NoRpcAdapter:
    @staticmethod
    def init(*_args, **_kwargs):
        return None

    @staticmethod
    def remote(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func


_NO_RPC_ADAPTER = _NoRpcAdapter()




@functools.lru_cache(None)
def _load_rpc() -> Any:
    try:
        # pylint:disable=import-outside-toplevel
        import db0_rpc

        return db0_rpc
    except ModuleNotFoundError:
        return _NO_RPC_ADAPTER


rpc = _load_rpc()


def has_rpc() -> bool:
    return rpc is not _NO_RPC_ADAPTER
