
import os
import random
import string
import tempfile

TEMP_FILENAME = os.path.join(tempfile.gettempdir(), "iotools_")


def _get_temp_filename(format=None):
    """
    Returns a temporary filename
    """
    if isinstance(format, str):
        format = "." + format.lower()
    else:
        format = ""
    for _ in range(10):
        temp_filename = TEMP_FILENAME + "".join(random.choice(string.ascii_letters) for i in range(8)) + format
        if not os.path.exists(temp_filename):
            return temp_filename
    raise OSError("Could not find a non-used temporary file")
