# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
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
