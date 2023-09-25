# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.

"""FFI APIs for AIPU Compass."""
import tvm


tvm._ffi._init_api("aipu_compass", __name__)
