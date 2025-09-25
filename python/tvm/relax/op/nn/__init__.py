# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Neural network related operators."""
#
# This file has been modified by Arm China team.
#
from .nn import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    attention,
    attention_bias,
    attention_var_len,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    batch_norm,
    batch_to_space_nd,
    conv1d,
    conv1d_transpose,
    conv2d,
    conv2d_transpose,
    conv3d,
    cross_entropy_with_logits,
    depth_to_space,
    dropout,
    gelu,
    gelu_tanh,
    group_norm,
    layer_norm,
    leakyrelu,
    log_softmax,
    lrn,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    nll_loss,
    pad,
    pixel_shuffle,
    prelu,
    relu,
    relu6,
    rms_norm,
    selu,
    silu,
    softmax,
    softplus,
    space_to_batch_nd,
    space_to_depth,
)
