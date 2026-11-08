# conftest.py
# 让 pytest 在项目根目录启动时能直接 import core / server / scripts。
# 之前 `pytest` 会把 tests/ 目录放到 sys.path 最前面，导致 `from server import ...` 失败。
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
