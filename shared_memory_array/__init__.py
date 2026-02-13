from .shared_memory_array import SharedMemoryArray, ManagedSharedMemoryArray

# Get the version from _version.py (added when building using scm)
try:
    from ._version import __version__ # noqa
except ModuleNotFoundError as e:
    __version__ = '0.0.0-dev'