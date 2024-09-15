# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from urllib.parse import urlparse
from typing import TypedDict, List, Union
import numpy as np
import os

__all__ = ['DataType', 'is_url_or_local']


# =========
# Data Type
# =========

class DataType(TypedDict):
    """
    Standard data type for input data.
    """

    X: Union[List[List[int]], np.ndarray[np.integer]]
    Y: Union[List[List[int]], np.ndarray[np.integer]]
    models: Union[List[str], np.ndarray[np.str_]]


# ===============
# is url or local
# ===============

def is_url_or_local(string):
    """
    Check a string is URL, a local file, or none.
    """

    parsed = urlparse(string)

    if parsed.scheme in ('http', 'https', 'ftp'):
        return 'url'

    elif os.path.exists(string):
        return 'local'

    return None
