from .client import Redis, StrictRedis
from .connection import (
    BlockingConnectionPool,
    ConnectionPool,
    Connection,
    SSLConnection,
    UnixDomainSocketConnection
)
from .utils import from_url
from .exceptions import (
    AuthenticationError,
    AuthenticationWrongNumberOfArgsError,
    BusyLoadingError,
    ChildDeadlockedError,
    ConnectionError,
    DataError,
    InvalidResponse,
    PubSubError,
    ReadOnlyError,
    RedisError,
    ResponseError,
    TimeoutError,
    WatchError
)


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


__version__ = '3.5.3'
VERSION = tuple(map(int_or_str, __version__.split('.')))

__all__ = [
    'AuthenticationError',
    'AuthenticationWrongNumberOfArgsError',
    'BlockingConnectionPool',
    'BusyLoadingError',
    'ChildDeadlockedError',
    'Connection',
    'ConnectionError',
    'ConnectionPool',
    'DataError',
    'from_url',
    'InvalidResponse',
    'PubSubError',
    'ReadOnlyError',
    'Redis',
    'RedisError',
    'ResponseError',
    'SSLConnection',
    'StrictRedis',
    'TimeoutError',
    'UnixDomainSocketConnection',
    'WatchError',
]
