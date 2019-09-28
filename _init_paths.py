# -*- coding: utf-8 -*-
"""
Append Faster-RCNN path to current path.

This is a separate file in order to avoid the
"module level import not at top of file" pep8 warning in main.py.
"""

import os
import sys

FASTER_RCNN_PATH = os.path.join(os.path.dirname(__file__), 'faster_rcnn')
if FASTER_RCNN_PATH not in sys.path:
    sys.path.append(FASTER_RCNN_PATH)
