try:
    # Fast, but only works in Python 3.8+.
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("seq2squiggle")
    except PackageNotFoundError:
        __version__ = None
except ImportError:
    # Slow, but works for all Python 3+.
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution("seq2squiggle").version
    except DistributionNotFound:
        __version__ = None

