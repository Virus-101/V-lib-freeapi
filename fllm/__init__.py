"""fllm: the short name for freeaipi.

    import fllm
    fllm.chat("hello").content

Everything in ``freeaipi`` is re-exported here, so the two names are
interchangeable. ``fllm`` is just fewer keystrokes.
"""

from freeaipi import *  # noqa: F401,F403
from freeaipi import __all__, __version__  # noqa: F401
