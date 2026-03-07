import functools
from typing import Any


class _RpcMock:
    @staticmethod
    def remote(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func


_RPC_MOCK = _RpcMock()




@functools.lru_cache(None)
def rpc() -> Any:
    try:
        # pylint:disable=import-outside-toplevel
        import db0_rpc

        return db0_rpc
    except ModuleNotFoundError:
        return _RPC_MOCK


def has_rpc() -> bool:
    return rpc() is not _RPC_MOCK
