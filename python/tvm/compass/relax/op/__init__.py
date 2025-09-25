# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Relax op contrib"""
import os
import sys
import importlib
import pkgutil
from .op import *
from .legalize import *
from .plugin import *
from .compute_count import *


# Scan and import all plugins.
plugin_paths = os.environ.get("AIPUPLUGIN_PATH", "").split(":") + ["./plugin", "."]
sys.path = plugin_paths + sys.path

ALL_MODULES = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules(plugin_paths)
    if name.startswith("compass_tvm_")
}
