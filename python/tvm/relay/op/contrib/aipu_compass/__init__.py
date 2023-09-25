# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""AIPU Compass extension of Relay Operator."""
import os
import sys
import pkgutil
import importlib
from .op import *
from .plugin import *
from . import strategy


# Scan and import all plugins.
AIPUPLUGIN_PATH = os.environ.get("AIPUPLUGIN_PATH", "").split(":") + ["./plugin", "."]
sys.path = AIPUPLUGIN_PATH + sys.path

ALL_MODULES = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules(AIPUPLUGIN_PATH)
    if name.startswith("aipu_tvm_")
}
