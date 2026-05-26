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
