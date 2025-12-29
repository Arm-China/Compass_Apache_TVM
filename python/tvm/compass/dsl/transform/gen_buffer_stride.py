# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Generate stride for autobroadcast buffer type."""
from tvm import tir
from tvm.tir.buffer import decl_buffer


def _gen_stride(shape):
    strides = []
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        expr = tir.if_then_else(shape[i] == 1, 0, stride)
        stride *= shape[i]
        strides.append(expr)
    strides.reverse()
    return strides


class _Rewriter(tir.StmtExprMutator):
    def __init__(self, buffer_mapping):
        super().__init__()
        self._buffer_mapping = buffer_mapping

    def _gen_buffer_stride(self, buffer_stmt):  # pylint: disable=unused-argument
        buffer = buffer_stmt.buffer
        if buffer.buffer_type != 2:
            return buffer_stmt

        if self._buffer_mapping.get(buffer.data) is not None:
            new_buffer = self._buffer_mapping[buffer.data]
        else:
            new_buffer = decl_buffer(
                buffer.shape,
                str(buffer.dtype),
                str(buffer.name),
                buffer.data,
                _gen_stride(buffer.shape),
                buffer.elem_offset,
                str(buffer.scope),
                buffer.data_alignment,
                buffer.offset_factor,
                "auto_broadcast",
                buffer.axis_separators,
                buffer.span,
            )

        if isinstance(buffer_stmt, tir.BufferLoad):
            return tir.BufferLoad(
                new_buffer, buffer_stmt.indices, buffer_stmt.predicate, buffer_stmt.span
            )
        else:
            return tir.BufferStore(
                new_buffer,
                buffer_stmt.value,
                buffer_stmt.indices,
                buffer_stmt.predicate,
                buffer_stmt.span,
            )

    def visit_buffer_load(self, buf_load):
        ret = super().visit_buffer_load(buf_load)
        return self._gen_buffer_stride(ret)

    def visit_buffer_store(self, buf_store):
        ret = super().visit_buffer_store(buf_store)
        return self._gen_buffer_stride(ret)


@tir.transform.prim_func_pass(opt_level=0)
class GenBufferStride:
    """Generate stride for autobroadcast buffer type."""

    def transform_function(self, f, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        is_autobroadcast = False
        buffer_map = f.buffer_map
        new_buffer_map = {}
        buffer_mapping = {}
        for k, v in buffer_map.items():
            # kAutoBroadcast == 2
            if v.buffer_type != 2:
                new_buffer_map[k] = v
            else:
                new_buffer = decl_buffer(
                    v.shape,
                    str(v.dtype),
                    str(v.name),
                    v.data,
                    _gen_stride(v.shape),
                    v.elem_offset,
                    str(v.scope),
                    v.data_alignment,
                    v.offset_factor,
                    "auto_broadcast",
                    v.axis_separators,
                    v.span,
                )
                new_buffer_map[k] = new_buffer
                buffer_mapping[v.data] = new_buffer
                is_autobroadcast = True

        if is_autobroadcast:
            f = tir.PrimFunc(f.params, f.body, f.ret_type, new_buffer_map, f.attrs, f.span)
        return f.with_body(_Rewriter(buffer_mapping).visit(f.body), span=f.span)
