# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name, import-outside-toplevel
"""Tensorflow lite frontend."""
import itertools
import math
from typing import Dict
import numpy as np
import tvm
from tvm import relax

try:
    from tflite.ActivationFunctionType import ActivationFunctionType
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.TensorType import TensorType
    from tflite.BuiltinOptions import BuiltinOptions
except ImportError:
    raise ImportError("The tflite package must be installed")


__all__ = ["from_tflite"]


def _to_int_list(np_array):
    """Convert a np array to a python int list.

    Note: This function converts np.int32 to python's int.
    If we don't do this conversion, numpy's automatic upcast will make
    the shape / parameters be converted to int64 IntImm in relax and
    cause problems in relax/TOPI.
    """
    return [int(x) for x in np_array]


class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


class TFLiteGraphImporter(object):
    """A helper class for handling Relax graph copying from TFLite GraphDef."""

    def __init__(self, model, keep_params_in_input):
        # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
        try:
            import tflite

            assert isinstance(model, tflite.Model)
        except TypeError:
            import tflite.Model

            assert isinstance(model, tflite.Model.Model)

        # keep the same as tflite
        assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"

        self.model = model
        self.subgraph = model.Subgraphs(0)
        self._keep_params_in_input = keep_params_in_input
        self._name2expr: Dict[str, relax.Expr] = {}
        self._params: Dict[relax.Var, tvm.nd.array] = {}
        self.bb: relax.BlockBuilder = relax.BlockBuilder()

        self.builtin_op_code = _build_str_map(BuiltinOperator())
        self.activation_fn_type = _build_str_map(ActivationFunctionType())

        # Add more operators
        self.convert_map = {
            "ADD": self.convert_add,
            "AVERAGE_POOL_2D": self.convert_average_pool2d,
            "CAST": self.convert_cast,
            "CONCATENATION": self.convert_concatenation,
            "CONV_2D": self.convert_conv2d,
            "DEPTH_TO_SPACE": self.convert_depth_to_space,
            "DEPTHWISE_CONV_2D": self.convert_depthwise_conv2d,
            "DEQUANTIZE": self.convert_dequantize,
            "EXP": self.convert_exp,
            "FULLY_CONNECTED": self.convert_fully_connected,
            "GATHER": self.convert_gather,
            "HARD_SWISH": self.convert_hard_swish,
            "LEAKY_RELU": self.convert_leaky_relu,
            "LOGISTIC": self.convert_logistic,
            "MAX_POOL_2D": self.convert_max_pool2d,
            "MEAN": self.convert_reduce_mean,
            "MINIMUM": self.convert_minimum,
            "MIRROR_PAD": self.convert_mirror_pad,
            "MUL": self.convert_mul,
            "PACK": self.convert_pack,
            "PAD": self.convert_pad,
            "PRELU": self.convert_prelu,
            "QUANTIZE": self.convert_quantize,
            "RELU": self.convert_relu,
            "RESHAPE": self.convert_reshape,
            "RESIZE_BILINEAR": self.convert_resize_bilinear,
            "RESIZE_NEAREST_NEIGHBOR": self.convert_resize_nearest_neighbor,
            "RSQRT": self.convert_rsqrt,
            "REVERSE_V2": self.convert_reverse_v2,
            "SOFTMAX": self.convert_softmax,
            "SPACE_TO_DEPTH": self.convert_space_to_depth,
            "SPLIT": self.convert_split,
            "SQUARED_DIFFERENCE": self.convert_squared_difference,
            "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": self.convert_sub,
            "TANH": self.convert_tanh,
            "TRANSPOSE_CONV": self.convert_transpose_conv,
            "TRANSPOSE": self.convert_transpose,
            "UNPACK": self.convert_unpack,
        }

    def from_tflite(self, shape_dict):
        """Construct Relax expressions from the tflite graph.

        Parameters
        ----------
        shape_dict : Dictionary of input dimensions
            Graph level input shape dictionary.

        Returns
        -------
        mod : tvm.IRModule
            The returned relax module
        """
        _shape_dict, _dtype_dict = _get_input_dict(self.model)
        if shape_dict is not None:
            _shape_dict.update(shape_dict)

        # model inputs / outputs
        model_inputs = self.subgraph.InputsAsNumpy()
        model_outputs = self.subgraph.OutputsAsNumpy()

        for model_input in model_inputs:
            in_name = _get_tensor_name(self.subgraph, model_input)
            shape = _shape_dict.get(in_name, None)
            assert shape is not None, f"The shape of {in_name} is not provided."
            dtype = _dtype_dict.get(in_name, "float32")
            self._name2expr[in_name] = relax.Var(in_name, relax.TensorStructInfo(shape, dtype))

        with self.bb.function("main"):
            with self.bb.dataflow():
                self._check_unsupported_ops()
                self._convert_op_to_relax()

                outputs = list()
                for n in model_outputs:
                    name = _get_tensor_name(self.subgraph, n)
                    outputs.append(self._name2expr[name])
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                output_var = self.bb.emit_output(outputs)

            # Create function attributes for this module
            func_attrs = {"num_input": len(model_inputs)}
            # Create a function from our output expression and all input variables.
            input_list = [self._name2expr[_get_tensor_name(self.subgraph, i)] for i in model_inputs]
            # Attach params if they are available.
            if self._keep_params_in_input and self._params:
                param_var_list, param_value_list = map(list, zip(*self._params.items()))
                input_list = input_list + param_var_list
                func_attrs["params"] = param_value_list

            self.bb.emit_func_output(output_var, params=input_list)

        relax_mod = self.bb.get()
        # Attach attributes.
        relax_mod["main"] = relax_mod["main"].with_attrs(func_attrs)
        return relax_mod

    def _check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()
        dynamic_range_ops_set = set()
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)
                continue

            # Trying to exclude "dynamic range quantization" optimized ops as not supported in TVM
            inputs = self.get_input_tensors(op)[0:1]
            weights = self.get_input_tensors(op)[1:]
            outputs = self.get_output_tensors(op)
            qnn_in_cnt = len([_.qnn_params for _ in inputs if _.qnn_params is not None])
            qnn_weight_cnt = len([_.qnn_params for _ in weights if _.qnn_params is not None])
            qnn_out_cnt = len([_.qnn_params for _ in outputs if _.qnn_params is not None])

            if qnn_in_cnt == 0 and qnn_out_cnt == 0 and qnn_weight_cnt > 0:
                dynamic_range_ops_set.add(op_code_str)

        raise_msg = ""

        if unsupported_ops_set:
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise_msg += f"The following operators are not supported in frontend TFLite: {ops}\n"

        if dynamic_range_ops_set:
            ops = str(list(dynamic_range_ops_set)).strip("[,]")
            raise_msg += (
                f"The following operators are likely to have dynamic range quantization: {ops}. "
                f"If you are running an optimized graph, please turn off dynamic range "
                f"quantization or use full integer quantization"
            )

        if len(raise_msg) > 0:
            raise tvm.error.OpNotImplemented(raise_msg)

    def _convert_op_to_relax(self):
        """Convert TFLite ops to relax ops"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            from tflite.Operator import Operator

            assert isinstance(op, Operator)
            ret = self.convert_map[op_code_str](op)
            ret = self.bb.normalize(ret)

            output_tensors = self.get_output_tensors(op)
            if len(output_tensors) == 1:
                tensor_idx = output_tensors[0].tensor_idx
                self._name2expr[_get_tensor_name(self.subgraph, tensor_idx)] = ret
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    name = _get_tensor_name(self.subgraph, output_tensor.tensor_idx)
                    self._name2expr[name] = ret[idx]

    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        op_code_list_idx = op.OpcodeIndex()

        op_c = self.model.OperatorCodes(op_code_list_idx)
        # In TFlite 2.4.x there was a change where the type of the field that contained
        # the builtin code changed from int8 to int32 in the flat buffer representation.
        # However, to retain support for old flat buffers that were created, they retained
        # the original 8 bit field, but named it "deprecated_builtin_code" in TFLite 2.4.
        # This means that the API function BuiltinCode() which originally returned the value
        # of the 8 bit field would now look for the value in the new int32 field in the
        # schema and DeprecatedBuiltinCode() will look at the old 8 bit field.
        # In TFLite 2.4, if the opcode value is less than 127, it can be in either field
        # (however, if it is only in the "builtin_code" field, the model is not backward
        # compatible), so similarly to TFLite 2.4 reader, we'll pick the higher value of the
        # two fields.
        # Remember however that this value came into existence only after Tensorflow
        # lite 2.4.x and hence encase it in a try -except block.
        # Phew !
        try:
            opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
        except AttributeError:
            # In versions before 2.4 the int8 field that holds the builtin code is accessed
            # by BuiltinCode() and DeprecatedBuiltinCode() doesn't exist
            opc = op_c.BuiltinCode()

        op_code_id = opc
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            msg = f"TFLite operator with code {str(op_code_id)}"
            msg += " is not supported by this version of the fbs schema."
            raise NotImplementedError(msg)
        if op_code_id == BuiltinOperator.CUSTOM:
            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        if not np.all(tflite_zero_point == 0):
                            msg = "TFLite per-axis quantization restricts all zero points to be"
                            msg += " 0, but a non-zero value is observed"
                            raise tvm.error.OpAttributeInvalid(msg)
                        zero_point = int(tflite_zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        msg = f"Quantized type {type(tflite_scale)} (scale) and  "
                        msg += f"{type(tflite_zero_point)} (zero point) not supported"
                        raise NotImplementedError(msg)
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(f"Quantized type {type(tflite_scale)} not supported")

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = relax.const(scale, "float32")
                    qnn_params["zero_point"] = relax.const(zero_point, "int32")
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except KeyError:
            msg = f"Tensor type '{tensor_wrapper.tensor.Type()}' currently not supported"
            raise NotImplementedError(msg)

    def get_tensor_value(self, tensor_wrapper, is_sparse=False):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = _to_int_list(self.get_tensor_shape(tensor_wrapper))
        else:
            shape = []

        if is_sparse:
            return np.frombuffer(data, dtype=dtype)
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.INT16:
            return "int16"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(f"Tensor type {str(tensor_type)} is currently not supported")

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params["scale"].data.numpy()
        rhs_scale = rhs_tensor.qnn_params["scale"].data.numpy()
        lhs_zero_point = lhs_tensor.qnn_params["zero_point"].data.numpy()
        rhs_zero_point = rhs_tensor.qnn_params["zero_point"].data.numpy()
        # 0.1 + 0.2 != 0.3
        return np.allclose(lhs_scale, rhs_scale, rtol=1e-5, atol=1e-5) and np.allclose(
            lhs_zero_point, rhs_zero_point, rtol=1e-5, atol=1e-5
        )

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

    def quantize(self, expr, tensor, axis=-1):
        """Helper function to quantize a tensor with Relax"""
        zp = tensor.qnn_params["zero_point"].data.numpy()
        if not zp.shape:
            zp = zp.item(0)
        zp_value = relax.const(zp, "int32")
        quantized = relax.op.quantize(
            data=expr,
            scale=tensor.qnn_params["scale"],
            zero_point=zp_value,
            axis=axis,
            out_dtype=self.get_tensor_type_str(tensor.tensor.Type()),
        )
        return quantized

    def dequantize(self, expr, tensor, axis=-1):
        """Helper function to dequantize a tensor with Relax"""
        zp = tensor.qnn_params["zero_point"].data.numpy()
        if not zp.shape:
            zp = zp.item(0)
        scale = tensor.qnn_params["scale"]
        if isinstance(expr, relax.Constant) and not expr.data.shape:
            expr = relax.const(expr.data.numpy().reshape([1]), expr.data.dtype)

        zp_value = relax.const(zp, "int32")
        dequantized = relax.op.dequantize(
            data=expr,
            scale=scale,
            axis=axis,
            zero_point=zp_value,
        )
        return dequantized

    def convert_qnn_fused_activation_function(
        self, expr, fused_activation_fn, scale, zero_point, dtype
    ):
        """Convert TFLite fused activation function. The expr is an input quantized tensor with
        scale and zero point"""
        # Quantize a float value to an quantized integer value
        quantize = lambda x: float(int(round(x / scale)) + zero_point)

        # Get min/max of the output dtype. This will be used to ensure that clip a_min/a_max are not
        # beyond the dtype range.
        qmin = float(tvm.tir.op.min_value(dtype).value)
        qmax = float(tvm.tir.op.max_value(dtype).value)
        int64_max = 2**63 - 1

        # The input expr is a quantized tensor with its scale and zero point. We calculate the
        # suitable clip off points based on these scale and zero point.
        if fused_activation_fn == ActivationFunctionType.NONE:
            return expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return relax.op.clip(expr, min=quantize(0), max=quantize(6.0))
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return relax.op.clip(expr, min=max(qmin, quantize(-1.0)), max=min(qmax, quantize(1.0)))
        if fused_activation_fn == ActivationFunctionType.RELU:
            return relax.op.clip(expr, min=max(qmin, quantize(0.0)), max=int64_max)

        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        msg = f"Quantized activation {fused_activation_fn_str} is not supported yet."
        raise tvm.error.OpNotImplemented(msg)

    def convert_conv2d(self, op):
        """Convert TFLite conv2d"""
        return self.convert_conv(op, "conv2d")

    def convert_depthwise_conv2d(self, op):
        """Convert TFLite depthwise conv2d"""
        return self.convert_conv(op, "depthwise")

    def convert_average_pool2d(self, op):
        """Convert TFLite average pool2d"""
        return self.convert_pool2d(op, "average")

    def convert_max_pool2d(self, op):
        """Convert TFLite max pool2d"""
        return self.convert_pool2d(op, "max")

    def convert_reshape(self, op):
        """Convert TFLite reshape"""
        from tflite.ReshapeOptions import ReshapeOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            if self.has_expr(shape_tensor.tensor_idx):
                raise NotImplementedError("Not supported yet.")
            target_shape = self.get_tensor_value(shape_tensor).tolist()
        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = _to_int_list(reshape_options.NewShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        # If the tensors are quantized, ensure that input/output qnn params are same.
        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        if input_tensor.qnn_params and input_tensor_type_str == "int8":
            # TFLite 2.x quantization spec requires qnn params to be same and dtype to be int8.
            # For TFLite 1.x, dtype can be uint8 and qnn params can be different
            output_tensor = output_tensors[0]
            msg = "TFLite reshape requires input and output scale and zero points to be equal"
            assert self.has_same_qnn_params(input_tensor, output_tensor), msg

        out = relax.op.reshape(in_expr, target_shape)
        if input_tensor.qnn_params and input_tensor_type_str == "uint8":
            output_tensor = output_tensors[0]
            if not self.has_same_qnn_params(input_tensor, output_tensor):
                out = self.dequantize(out, input_tensor)
                out = self.quantize(out, output_tensor)

        return out

    def _convert_resize(self, method, op):
        """Generic method to Convert TFLite RESIZE operators"""
        from tflite.ResizeBilinearOptions import ResizeBilinearOptions

        # ResizeNearestNeighborOptions was added in tflite v1.13
        tflite_ver = 1120
        if "ResizeNearestNeighborOptions" in dir(BuiltinOptions):
            from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions

            tflite_ver = 1130

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # images, 4-D Tensor with shape NHWC.
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # size - 1-D int32 Tensor of 2 elements: new_height, new_width
        target_size = tuple(self.get_tensor_value(input_tensors[1]))

        # Options - align_corners (bool)
        resize_options = None
        align_corners = False
        bilinear_method = method == "linear"
        if bilinear_method:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeBilinearOptions
            resize_options = ResizeBilinearOptions()
        elif tflite_ver >= 1130:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeNearestNeighborOptions
            resize_options = ResizeNearestNeighborOptions()

        if resize_options is not None:
            op_options = op.BuiltinOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = resize_options.AlignCorners()
            half_pixel_centers = resize_options.HalfPixelCenters()

        # Use layout NHWC
        coord_trans = "asymmetric"
        rounding_method = "floor"
        if align_corners:
            coord_trans = "align_corners"
            rounding_method = "round_prefer_ceil"
        if half_pixel_centers:
            if method == "nearest_neighbor":
                coord_trans = "tf_half_pixel_for_nn"
            else:
                coord_trans = "half_pixel"

        if bilinear_method and input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.image.resize2d(
            in_expr, target_size, None, "NHWC", method, coord_trans, rounding_method
        )
        if bilinear_method and output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_resize_bilinear(self, op):
        """Convert TFLite RESIZE_BILINEAR"""
        return self._convert_resize("linear", op)

    def convert_resize_nearest_neighbor(self, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR"""
        return self._convert_resize("nearest_neighbor", op)

    def convert_logistic(self, op):
        """Convert TFLite LOGISTIC"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.sigmoid(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        params = {"axis": -1}  # -1 is channel
        in_expr = self.get_expr(input_tensor_idx)

        # TODO - Naive softmax int8 implementation leads to bad accuracy. Currently, we can
        # dequantize to FP32 and perform softmax on FP32. We can investigate an integer only softmax
        # implementation in future.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = relax.op.nn.softmax(in_expr, **params)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_tanh(self, op):
        """Convert TFLite TANH"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.tanh(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_relu(self, op):
        """Convert TFLite ReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.nn.relu(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_hard_swish(self, op):
        """Convert TFLite Hard swish"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        def _relu6(data):
            return relax.op.clip(data, 0.0, 6.0)

        def _hard_swish(data):
            return data * _relu6(data + relax.const(3.0)) / relax.const(6.0)

        # Dequantize if the input is quantized.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        # Perform hardswish
        out = _hard_swish(in_expr)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_leaky_relu(self, op):
        """Convert TFLite LEAKY_RELU"""
        from tflite.LeakyReluOptions import LeakyReluOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.LeakyReluOptions
        op_options = op.BuiltinOptions()
        leaky_relu_options = LeakyReluOptions()
        leaky_relu_options.Init(op_options.Bytes, op_options.Pos)
        alpha_tensor = leaky_relu_options.Alpha()

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.nn.leakyrelu(in_expr, alpha_tensor)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_concatenation(self, op):
        """Convert TFLite concatenation"""
        from tflite.ConcatenationOptions import ConcatenationOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concatenation_options = ConcatenationOptions()
        concatenation_options.Init(op_options.Bytes, op_options.Pos)
        concatenation_axis = concatenation_options.Axis()
        fused_activation_fn = concatenation_options.FusedActivationFunction()

        if not input_tensors[0].qnn_params:
            out = relax.op.concat(in_exprs, axis=concatenation_axis)
        else:
            in_exprs_fp32 = list()
            for idx, in_expr in enumerate(in_exprs):
                in_exprs_fp32.append(self.dequantize(in_expr, input_tensors[idx]))
            out_fp32 = relax.op.concat(in_exprs_fp32, concatenation_axis)
            out = self.quantize(out_fp32, output_tensor, axis=concatenation_axis)

        # Handle fused activations
        if output_tensor.qnn_params:
            scale_val = _get_scalar_from_constant(output_tensor.qnn_params["scale"])
            zero_point_val = _get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def _convert_unary_elemwise(self, relax_op, op):
        """Generic method to convert TFLite unary elemwise functions"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax_op(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_exp(self, op):
        """Convert TFLite EXP"""
        return self._convert_unary_elemwise(relax.op.exp, op)

    def convert_rsqrt(self, op):
        """Convert TFLite RSQRT"""
        return self._convert_unary_elemwise(relax.op.rsqrt, op)

    def _convert_elemwise(self, relax_op, op):
        """Generic method to Convert TFLite elemwise"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_expr = self.get_tensor_expr(rhs_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # If quantized, extracts qnn params and call dequant+op+quant.
        if lhs_tensor.qnn_params:
            assert rhs_tensor.qnn_params, "Both tensors should be quantized."
            assert output_tensor.qnn_params, "Output tensor should be quantized."
            lhs_expr_f32 = self.dequantize(lhs_expr, lhs_tensor)
            rhs_expr_f32 = self.dequantize(rhs_expr, rhs_tensor)
            out = relax_op(lhs_expr_f32, rhs_expr_f32)
        else:
            out = relax_op(lhs_expr, rhs_expr)

        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_add(self, op):
        """Convert TFLite ADD"""
        return self._convert_elemwise(relax.op.add, op)

    def convert_sub(self, op):
        """Convert TFLite SUB"""
        return self._convert_elemwise(relax.op.subtract, op)

    def convert_mul(self, op):
        """Convert TFLite MUL"""
        return self._convert_elemwise(relax.op.multiply, op)

    def convert_minimum(self, op):
        """Convert TFLite MINIMUM"""
        return self._convert_elemwise(relax.op.minimum, op)

    def convert_squared_difference(self, op):
        """Convert TFLite SQUARED DIFFERENCE"""
        # Check if the input tensor is quantized, call QNN op
        # (https://github.com/tensorflow/tflite-micro/blob/bc35c3ed9c7ab93b3a13b46fce936f854bcfce2c/tensorflow/lite/micro/kernels/squared_difference.cc#L157)  # pylint: disable=line-too-long
        assert self.is_quantized(op)

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        lhs_expr = self.get_tensor_expr(input_tensors[0])
        rhs_expr = self.get_tensor_expr(input_tensors[1])
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(output_tensors) == 1, "output tensors length should be 1"
        lhs_expr_f32 = self.dequantize(lhs_expr, input_tensors[0])
        rhs_expr_f32 = self.dequantize(rhs_expr, input_tensors[1])
        out_f32 = relax.op.subtract(lhs_expr_f32, rhs_expr_f32)
        return self.quantize(out_f32 * out_f32, output_tensors[0])

    def convert_gather(self, op):
        """Method to Convert TFLite GATHER operator"""
        from tflite.GatherOptions import GatherOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        data = self.get_tensor_expr(input_tensors[0])
        indices = input_tensors[1]
        indices_type = indices.tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)

        assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
        op_options = op.BuiltinOptions()
        gather_options = GatherOptions()
        gather_options.Init(op_options.Bytes, op_options.Pos)
        axis = gather_options.Axis()

        # Check the indices are with in bounds.
        data_shape = _to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)

        axis = data_dim + axis if axis < 0 else axis
        assert 0 <= axis < data_dim, "Axis out of bounds"

        if self.has_expr(indices.tensor_idx):
            indices_expr = self.get_expr(indices.tensor_idx)
            if str(indices_expr.struct_info.dtype) != "int32":
                indices_expr = relax.op.astype(indices_expr, "int32")
        else:
            indices_val = self.get_tensor_value(indices)
            indices_expr = self.exp_tab.new_const(
                indices_val,
                dtype=self.get_tensor_type_str(indices_type),
                source_name=indices.tensor.Name(),
            )
            indices_shape = list(indices_val.shape)
            indices_len = len(indices_shape)

            out_shape = data_shape[:axis] + indices_shape[:] + data_shape[axis + 1 :]

            loopover = [range(s) for s in out_shape]
            for idx in list(itertools.product(*loopover)):
                real_indices = (
                    list(idx[:axis])
                    + [indices_val[idx[axis : axis + indices_len]]]
                    + list(idx[axis + indices_len :])
                )
                if np.any(np.subtract(data_shape, real_indices) < 0):
                    raise ValueError("TFLite out of bound indices are not supported.")

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # If quantized, extracts qnn params and call dequant+op+quant.
        if input_tensors[0].qnn_params:
            assert output_tensor.qnn_params, "Output tensor should be quantized."
            data_f32 = self.dequantize(data, input_tensors[0])
            out = relax.op.take(data_f32, indices_expr, axis=axis)
        else:
            out = relax.op.take(data, indices_expr, axis=axis)

        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_strided_slice(self, op):
        """Method to Convert TFLite STRIDED_SLICE operator.
        NOTE: Eventhough tensorflow supports begin_mask, end_mask, ellipsis_mask, new_axis_mask
        and shrink_axis_mask, tflite doesn't support these and expect these values to be zero.
        But in future, they may open up the mask implementation, so kept the implementation
        same as tensorflow.

        This op extracts a slice of size (end - begin) / stride from the given input tensor.
        Starting at the location specified by begin the slice continues by adding stride to the
        index until all dimensions are not less than end. Note that a stride can be negative,
        which causes a reverse slice.

        For slice input[val0, val1, ..., valn], begin/end/strides will be vectors of length n.

        In each mask field(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
        the ith bit will correspond to the ith val.

        If the ith bit of begin_mask is set, begin[i] is ignored and the fullest possible range
        in that dimension is used instead.

        If the ith bit of ellipsis_mask is set, as many unspecified dimensions as needed will be
        inserted between other dimensions. Only one non-zero bit is allowed in ellipsis_mask.

        If the ith bit of new_axis_mask is set, then begin, end, and stride are ignored and a
        new length 1 dimension is added at this point in the output tensor.

        If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks
        the dimensionality by 1, taking on the value at index begin[i]. end[i] and strides[i]
        are ignored in this case.
        begin and end are zero-indexed. strides entries must be non-zero.

        TVM Relax implementation of doesn't support mask, so the mask values are processed in
        this function and begin/end/strides are updated accordingly. If any mask is present, and
        since tvm doesn't support mask computation directly, the output need a final reshape.
        """
        from tflite.StridedSliceOptions import StridedSliceOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "input tensors length should be 4"

        data_expr = self.get_expr(input_tensors[0].tensor_idx)

        begin = list(self.get_tensor_value(input_tensors[1]))
        end = list(self.get_tensor_value(input_tensors[2]))
        stride = list(self.get_tensor_value(input_tensors[3]))

        assert op.BuiltinOptionsType() == BuiltinOptions.StridedSliceOptions
        op_options = op.BuiltinOptions()
        options = StridedSliceOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        begin_mask = options.BeginMask()
        end_mask = options.EndMask()
        ellipsis_mask = options.EllipsisMask()
        new_axis_mask = options.NewAxisMask()
        shrink_axis_mask = options.ShrinkAxisMask()

        data_shape = _to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)
        stride_dim = len(stride)

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 else data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = m_begin[final_index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)

        axes = list(range(len(begin)))

        def get_prim_value_list(values):
            new_values = []
            for v in values:
                new_values.append(relax.PrimValue(int(v)))
            return tuple(new_values)

        begin = get_prim_value_list(begin)
        end = get_prim_value_list(end)
        strides = get_prim_value_list(stride)
        out = relax.op.strided_slice(data_expr, axes=axes, begin=begin, end=end, strides=strides)
        out = self.bb.normalize(out)
        out_shape = out.struct_info.shape
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        # Create final output shape.
        final_output = []
        final_len = len(fshape_indices)
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
                final_len += 1
            elif gather_index == -2:
                final_len -= 1
            else:
                final_output.append(out_shape[gather_index])

        if final_len == 0:
            return relax.op.squeeze(out, axis=tuple(range(len(fshape_indices))))

        if not final_output:
            return out
        return relax.op.reshape(out, shape=tuple(final_output))

    def convert_reduce_mean(self, op):
        """Generic method to Convert TFLite REDUCE operators"""
        from tflite.ReducerOptions import ReducerOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # input_tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # axis
        axis_value = self.get_tensor_value(input_tensors[1])
        axis = tuple(axis_value) if len(axis_value.shape) > 0 else tuple((axis_value.item(),))

        # Options - keep_dims (bool)
        # In case Options are not present, set keep_dims to False(default)
        if op.BuiltinOptionsType():
            assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
            reduce_options = ReducerOptions()
            op_options = op.BuiltinOptions()
            reduce_options.Init(op_options.Bytes, op_options.Pos)
            keep_dims = reduce_options.KeepDims()
        else:
            keep_dims = False

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = relax.op.mean(in_expr, axis, keep_dims)

        # Finally if the reduce is quantized. Add a requantize at the end.
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_fully_connected(self, op):
        """Convert TFLite fully connected"""
        from tflite.FullyConnectedOptions import FullyConnectedOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (2, 3), "input tensors length should be two or three"

        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        weight_tensor_shape = _to_int_list(self.get_tensor_shape(weight_tensor))
        # Weight should have only 2 dimensions(TFLite convention)
        assert len(weight_tensor_shape) == 2, "Weight should be only 2-dim"

        target_shape = tuple((-1, weight_tensor_shape[1]))
        in_expr = self.get_tensor_expr(input_tensor)
        in_expr = relax.op.reshape(in_expr, target_shape)

        # TODO: Change the output shape calculation based on keep_dim option
        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()
        keep_num_dims = fully_connected_options.KeepNumDims()

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
            weight_expr = relax.op.permute_dims(weight_expr, axes=(1, 0))
        else:
            weight_value = self.get_tensor_value(weight_tensor)
            weight_value = np.transpose(weight_value, (1, 0))
            weight_expr = self._new_const_or_var(weight_value, weight_tensor_type_str)

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
            weight_expr = self.dequantize(weight_expr, weight_tensor)
        out = relax.op.matmul(in_expr, weight_expr)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            if bias_tensor.tensor_idx != -1:
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (quantization) or FLOAT32
                assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
                bias_expr = self.get_tensor_expr(bias_tensor)
                if bias_tensor.qnn_params:
                    bias_expr = self.dequantize(bias_expr, bias_tensor)
                out = relax.op.add(out, bias_expr)

        out = self.convert_fused_activation_function(out, fused_activation_fn)
        # Finally if the matmul is quantized. Add a quantize at the end.
        if output_tensors[0].qnn_params:
            out = self.quantize(out, output_tensors[0])

        # Change the output shape calculation based on keep_dim option
        if keep_num_dims:
            input_shape = self.get_tensor_expr(input_tensor).struct_info.shape.values
            output_shape = input_shape[:-1] + [weight_tensor_shape[0]]
            out = relax.op.reshape(out, tuple(output_shape))

        return out

    def convert_fused_activation_function(self, in_expr, fused_activation_fn):
        """Convert TFLite fused activation function"""
        if fused_activation_fn == ActivationFunctionType.NONE:
            return in_expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return relax.op.clip(in_expr, min=0, max=6)
        if fused_activation_fn == ActivationFunctionType.RELU:
            return relax.op.nn.relu(in_expr)
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return relax.op.clip(in_expr, min=-1, max=1)
        if fused_activation_fn == ActivationFunctionType.TANH:
            return relax.op.tanh(in_expr)
        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise tvm.error.OpNotImplemented(
            f"Fused activation {fused_activation_fn_str} is not supported yet."
        )

    def convert_conv(self, op, conv_type):
        """convolution implementation."""
        from tflite.Conv2DOptions import Conv2DOptions
        from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
        from tflite.Padding import Padding

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        is_depthwise_conv = False
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "depthwise":
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            msg = f"Operator {conv_type} is not supported for frontend TFLite."
            raise tvm.error.OpNotImplemented(msg)

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        input_tensor = input_tensors[0]
        _, input_h, input_w, input_c = _to_int_list(self.get_tensor_shape(input_tensor))

        weight_tensor = input_tensors[1]
        _, kernel_h, kernel_w, in_channels = _to_int_list(self.get_tensor_shape(weight_tensor))
        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            assert in_channels == input_c * depth_multiplier

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            "strides": [stride_h, stride_w],
            "dilation": [dilation_h, dilation_w],
            "padding": [0, 0],
            "data_layout": "NHWC",
        }

        if is_depthwise_conv:
            params["groups"] = int(input_c)
            # If number of input channels is 1, treat as normal convolution.
            params["kernel_layout"] = "HWIO" if input_c == 1 else "HWOI"
        else:
            params["kernel_layout"] = "HWIO"
            if input_c != in_channels:
                msg = "Input channels is not divisible of kernel in_channels."
                assert input_c % in_channels == 0, msg
                params["groups"] = int(input_c / in_channels)

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_tensor_expr(input_tensor)

        # TFLite kernel layout:
        # convolution:
        # OC KH KW IC, we require KH KW IC OC (HWIO)
        # depthwise convolution:
        # 1 KH KW C(input_c * depth_multiplier), we require KH KW IC M (depth_multiplier) (HWOI)
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
            if is_depthwise_conv:
                target_shape = (kernel_h, kernel_w, input_c, depth_multiplier)
                weight_expr = relax.op.reshape(weight_expr, target_shape)
            else:
                weight_expr = relax.op.permute_dims(weight_expr, axes=(1, 2, 3, 0))
        else:
            weight_value = self.get_tensor_value(weight_tensor)
            if is_depthwise_conv:
                target_shape = (kernel_h, kernel_w, input_c, depth_multiplier)
                weight_value = weight_value.reshape(target_shape)
            else:
                weight_value = weight_value.transpose((1, 2, 3, 0))
            weight_expr = self._new_const_or_var(weight_value, weight_tensor_type_str)

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = _get_pad_value(input_h, dilated_kernel_h, stride_h)
            pad_left, pad_right = _get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == pad_bottom == pad_left == pad_right == 0)
            if do_pad:
                params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            msg = f"Padding format {padding} is not supported for operator Conv."
            raise tvm.error.OpAttributeUnImplemented(msg)

        output_tensor = output_tensors[0]
        if input_tensor.qnn_params:
            in_expr_f32 = self.dequantize(in_expr, input_tensor)
            axis = 2 if is_depthwise_conv else -1
            w_expr_f32 = self.dequantize(weight_expr, weight_tensor, axis=axis)
            out = relax.op.nn.conv2d(in_expr_f32, w_expr_f32, **params)
        else:
            out = relax.op.nn.conv2d(in_expr, weight_expr, **params)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
            bias_dtype = self.get_tensor_type_str(bias_tensor_type)
            bias_expr = self.get_tensor_expr(bias_tensor, bias_dtype)
            if bias_tensor.qnn_params:
                b_expr_f32 = self.dequantize(bias_expr, bias_tensor)
                out = relax.op.add(out, b_expr_f32)
            else:
                out = relax.op.add(out, bias_expr)

        # Handle activation.
        out = self.convert_fused_activation_function(out, fused_activation_fn)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor, axis=3)
        return out

    def convert_split(self, op):
        """split implementation."""
        from tflite.SplitOptions import SplitOptions

        input_tensors = self.get_input_tensors(op)

        assert len(input_tensors) == 2, "input tensors length should be == 2"

        axis_tensor = input_tensors[0]
        split_axis = self.get_tensor_value(axis_tensor)
        input_tensor = input_tensors[1]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.SplitOptions
        op_options = op.BuiltinOptions()
        split_options = SplitOptions()
        split_options.Init(op_options.Bytes, op_options.Pos)
        num_splits = split_options.NumSplits()

        in_expr = self.get_expr(input_tensor_idx)
        out = relax.op.split(in_expr, num_splits, axis=int(split_axis))
        if isinstance(self.bb.normalize(out).struct_info, relax.TensorStructInfo):
            return in_expr
        return out

    def convert_transpose(self, op):
        """transpose implementation."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        # axis
        in_axis = tuple(self.get_tensor_value(input_tensors[1]))

        if not in_axis:
            out = relax.op.permute_dims(in_expr)
        else:
            out = relax.op.permute_dims(in_expr, in_axis)

        return out

    def convert_cast(self, op):
        """Convert TFLite CAST"""
        from tflite.CastOptions import CastOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # MLIR-based converter outputs no BuiltinOptions for Cast operator. In this
        # case the output type can be derived from the Cast operator output tensor.
        # When TOCO converter is used there will be "normal" BuiltinOptions.CastOptions
        # with output type.
        if op.BuiltinOptions() is not None:
            assert op.BuiltinOptionsType() == BuiltinOptions.CastOptions
            op_options = op.BuiltinOptions()
            cast_options = CastOptions()
            cast_options.Init(op_options.Bytes, op_options.Pos)
            cast_dtype = cast_options.OutDataType()
        else:
            cast_dtype = self.get_output_tensors(op)[0].tensor.Type()

        out = relax.op.astype(in_expr, self.get_tensor_type_str(cast_dtype))

        return out

    def convert_pool2d(self, op, pool_type):
        """pool2d implementation."""
        from tflite.Padding import Padding
        from tflite.Pool2DOptions import Pool2DOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        params = {
            "pool_size": (filter_h, filter_w),
            "strides": (stride_h, stride_w),
            "padding": [0, 0],
            "layout": "NHWC",
        }

        in_expr = self.get_expr(input_tensor_idx)
        if input_tensor.qnn_params:
            msg = "TFlite pool2d requires input and output qnn params to be same"
            assert self.has_same_qnn_params(input_tensor, output_tensor), msg
            in_expr = self.dequantize(in_expr, input_tensor)

        _, input_h, input_w, _ = _to_int_list(self.get_tensor_shape(input_tensor))

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = _get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = _get_pad_value(input_w, filter_w, stride_w)
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            msg = f"Padding format {padding} for operator Pool2D is not supported."
            raise tvm.error.OpAttributeUnImplemented(msg)

        if pool_type == "average":
            out = relax.op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            out = relax.op.nn.max_pool2d(in_expr, **params)
        else:
            msg = f"Operator {pool_type} pool is not supported for frontend TFLite."
            raise tvm.error.OpNotImplemented(msg)

        # Handle fused activations
        out = self.convert_fused_activation_function(out, fused_activation_fn)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_pad(self, op):
        """Convert TFLite PAD/PADV2 \
           TFLite treats PAD and PADV2 operators identically"""

        input_tensors = self.get_input_tensors(op)

        # TFLite PAD/PADV2 only supports CONSTANT mode
        msg = "input tensor's length should be 2 for PAD and 3 for PADV2"
        assert len(input_tensors) == 2 or len(input_tensors) == 3, msg

        if len(input_tensors) == 3:
            msg = "constant_values tensor must be of same type as input tensor"
            assert input_tensors[0].tensor.Type() == input_tensors[2].tensor.Type(), msg

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])

        # convert list of lists to tuple of tuples
        paddings = [item for sublist in pad_list for item in sublist]

        # Set the pad value, by default 0, unless constant_values parameter is provided
        pad_value = 0

        if input_tensor.qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            output_tensor = output_tensors[0]
            msg = "TFLite PADV2 requires input and output scale and zero points to be equal"
            assert self.has_same_qnn_params(input_tensor, output_tensor), msg

            # The pad value for quantized pad is the input zero point by default.
            pad_value = float(input_tensor.qnn_params["zero_point"].data.numpy())

        if len(input_tensors) == 3:
            pad_value = self.get_tensor_value(input_tensors[2])
            if isinstance(pad_value, np.ndarray):
                pad_value = pad_value.tolist()
            if isinstance(pad_value, list):
                assert len(pad_value) == 1, "Only one constant value is expected."
                pad_value = pad_value[0]
            if input_tensor.qnn_params:
                # Check that input tensor and constant_values have same qnn params.
                msg = "TFLite PADV2 requires input and constant_values tensors'"
                msg += "scale and zero points to be equal"
                assert self.has_same_qnn_params(input_tensor, input_tensors[2]), msg

        out = relax.op.nn.pad(in_expr, pad_width=paddings, pad_value=pad_value)
        return out

    def convert_mirror_pad(self, op):
        """Convert TFLite MIRROR_PAD"""
        from tflite.MirrorPadOptions import MirrorPadOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        output_tensor = output_tensors[0]

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        paddings = [item for sublist in pad_list for item in sublist]

        assert op.BuiltinOptionsType() == BuiltinOptions.MirrorPadOptions
        op_options = op.BuiltinOptions()
        mirror_pad_options = MirrorPadOptions()
        mirror_pad_options.Init(op_options.Bytes, op_options.Pos)
        mode_byte = mirror_pad_options.Mode()

        assert mode_byte == 0, "Only supports reflect mode right now."
        out = relax.op.nn.pad(in_expr, paddings, "reflect")

        if input_tensor.qnn_params and output_tensor.qnn_params:
            out_fp32 = self.dequantize(out, input_tensor)
            out = self.quantize(out_fp32, output_tensor)

        return out

    def convert_pack(self, op):
        """Convert TFLite pack"""
        from tflite.PackOptions import PackOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        if input_tensors[0].qnn_params:
            msg = "TFLite pack requires input and output scale and zero points to be equal"
            assert self.has_same_qnn_params(input_tensors[0], output_tensors[0]), msg

            for input_tensor in input_tensors:
                msg = "TFLite pack requires all input tensors to have same scale and zero point"
                assert self.has_same_qnn_params(input_tensors[0], input_tensor), msg

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        pack_axis = pack_options.Axis()
        pack_values_count = pack_options.ValuesCount()
        assert len(input_tensors) == pack_values_count, "Discordance in input values count"

        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]
        in_exprs_reshaped = [relax.op.expand_dims(_, axis=pack_axis) for _ in in_exprs]
        out = relax.op.concat(in_exprs_reshaped, pack_axis)
        return out

    def convert_unpack(self, op):
        """Convert TFLite unpack"""
        from tflite.UnpackOptions import UnpackOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        assert op.BuiltinOptionsType() == BuiltinOptions.UnpackOptions
        op_options = op.BuiltinOptions()
        unpack_options = UnpackOptions()
        unpack_options.Init(op_options.Bytes, op_options.Pos)
        num_unpacks = unpack_options.Num()
        unpack_axis = unpack_options.Axis()
        axis = [unpack_axis]
        assert num_unpacks != 1, "Currently only supports num_unpacks != 1"
        splitted = relax.op.split(in_expr, indices_or_sections=num_unpacks, axis=unpack_axis)
        op = self.bb.normalize(splitted)

        tuple_items = []
        for i in range(len(op.checked_type.fields)):
            tuple_items.append(self.bb.emit(relax.TupleGetItem(op, i)))
        op = relax.Tuple(tuple_items)

        return relax.Tuple([relax.op.squeeze(split_item, axis=axis) for split_item in op])

    def convert_depth_to_space(self, op):
        """Convert TFLite DEPTH_TO_SPACE"""
        from tflite.DepthToSpaceOptions import DepthToSpaceOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.DepthToSpaceOptions
        op_options = op.BuiltinOptions()
        depth_to_space_options = DepthToSpaceOptions()
        depth_to_space_options.Init(op_options.Bytes, op_options.Pos)
        block_size = depth_to_space_options.BlockSize()
        return relax.op.nn.depth_to_space(in_expr, block_size, layout="NHWC")

    def convert_space_to_depth(self, op):
        """Convert TFLite SPACE_TO_DEPTH"""
        from tflite.SpaceToDepthOptions import SpaceToDepthOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.SpaceToDepthOptions
        op_options = op.BuiltinOptions()
        space_to_depth_options = SpaceToDepthOptions()
        space_to_depth_options.Init(op_options.Bytes, op_options.Pos)
        block_size = space_to_depth_options.BlockSize()
        out = relax.op.nn.space_to_depth(in_expr, block_size, layout="NHWC")

        return out

    def convert_prelu(self, op):
        """Convert TFLite PReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_tensor = input_tensors[0]
        alpha_tensor = input_tensors[1]
        if self.has_expr(alpha_tensor.tensor_idx):
            raise NotImplementedError("Currently only supports alpha as constant value.")

        alpha_value = self.get_tensor_value(alpha_tensor)
        alpha_value = alpha_value.reshape([-1])

        alpha_tensor_type_str = self.get_tensor_type_str(alpha_tensor.tensor.Type())
        alpha_expr = self._new_const_or_var(alpha_value, alpha_tensor_type_str)

        in_expr = self.get_expr(input_tensor.tensor_idx)
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
            alpha_expr = self.dequantize(alpha_expr, alpha_tensor)
            output_tensor = self.get_output_tensors(op)[0]
            out_fp32 = relax.op.nn.prelu(in_expr, alpha_expr, axis=-1)
            out = self.quantize(out_fp32, output_tensor)
        else:
            out = relax.op.nn.prelu(in_expr, alpha_expr, axis=-1)
        return out

    def convert_transpose_conv(self, op):
        """Convert TFLite TRANSPOSE_CONV"""
        from tflite.Padding import Padding
        from tflite.TransposeConvOptions import TransposeConvOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 3, "input tensors length should be >= 3"

        # Input (data) Tensor. NHWC layout
        input_tensor = input_tensors[2]
        _, _, _, input_c = _to_int_list(self.get_tensor_shape(input_tensor))
        # Weights tensor. TFLite uses OHWI layout
        weights_tensor = input_tensors[1]
        out_channels, kernel_h, kernel_w, in_channels = _to_int_list(
            self.get_tensor_shape(weights_tensor)
        )

        msg = "Input channel in the filter should match to channel in the input"
        assert input_c == in_channels, msg
        # output_shape Tensor. NHWC layout
        output_shape_tensor = input_tensors[0]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.TransposeConvOptions
        op_options = op.BuiltinOptions()
        deconv_options = TransposeConvOptions()
        deconv_options.Init(op_options.Bytes, op_options.Pos)

        padding = deconv_options.Padding()
        stride_h = deconv_options.StrideH()
        stride_w = deconv_options.StrideW()
        msg = f"Padding format {padding} is not supported for operator TRANSPOSE_CONV"
        assert padding in (Padding.VALID, Padding.SAME), msg

        # Data
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # Weights
        weights_tensor_type = weights_tensor.tensor.Type()
        # weights tensor type should be UINT8 (quantization) or FLOAT32
        assert weights_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weights_tensor_type)

        if self.has_expr(weights_tensor.tensor_idx):
            weight_expr_iohw = self.get_expr(weights_tensor.tensor_idx)
            weight_expr_iohw = relax.op.permute_dims(weight_expr_iohw, axes=(3, 0, 1, 2))
        else:
            weight_value_ohwi = self.get_tensor_value(weights_tensor)
            # Relax kernel_layout should be OIHW
            # Relax weights layout should be different from kernel_layout - it should be IOHW
            weight_value_iohw = np.transpose(weight_value_ohwi, (3, 0, 1, 2))
            weight_expr_iohw = self._new_const_or_var(weight_value_iohw, weight_tensor_type_str)

        # Output shape value
        output_shape_value = self.get_tensor_value(output_shape_tensor)
        # Relax expects filter output channel to match to output tensor channel.
        msg = "Output channel in the filter should match to channel in the output_shape"
        assert out_channels == output_shape_value[3], msg

        if padding == Padding.SAME:
            output_h, output_w = output_shape_value[1], output_shape_value[2]
            pad_top, pad_bottom = _get_pad_value(output_h, kernel_h, stride_h)
            pad_left, pad_right = _get_pad_value(output_w, kernel_w, stride_w)
            padding = (pad_top, pad_left, pad_bottom, pad_right)
        else:
            padding = (0, 0, 0, 0)

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
            weight_expr_iohw = self.dequantize(weight_expr_iohw, weights_tensor, 1)

        out = relax.op.nn.conv2d_transpose(
            in_expr,
            weight_expr_iohw,
            strides=(stride_h, stride_w),
            padding=padding,
            data_layout="NHWC",
            kernel_layout="IOHW",
        )

        # Checking if there is a fused bias
        if len(input_tensors) == 4:
            bias_tensor = input_tensors[3]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
            bias_dtype = self.get_tensor_type_str(bias_tensor_type)
            bias_expr = self.get_tensor_expr(bias_tensor, bias_dtype)
            if bias_tensor.qnn_params:
                bias_expr = self.dequantize(bias_expr, bias_tensor)
            out = relax.op.add(out, bias_expr)

        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor, axis=3)

        return out

    def convert_quantize(self, op):
        """Convert TFLite Quantize"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        in_expr = self.get_tensor_expr(input_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # The output must be quantized
        assert output_tensor.qnn_params

        # TFLite Quantize op can also act as Requantize op
        if input_tensor_type_str == "float32":
            out = self.quantize(in_expr, output_tensor)
        else:
            in_expr_f32 = self.dequantize(in_expr, input_tensor)
            out = self.quantize(in_expr_f32, output_tensor)
        return out

    def convert_dequantize(self, op):
        """Convert TFLite Dequantize"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        msg = "Currently only supports dtype of input_tensor != float16"
        assert input_tensor.tensor.Type() != TensorType.FLOAT16, msg

        in_expr = self.get_expr(input_tensor.tensor_idx)

        # The input must be quantized
        assert input_tensor.qnn_params
        # Dequantize the input.
        out = self.dequantize(in_expr, input_tensor)

        return out

    def convert_reverse_v2(self, op):
        """Convert TFLite REVERSE_V2"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensor's length should be 2"

        input_expr = self.get_expr(input_tensors[0].tensor_idx)

        # Getting axis value
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert len(axis) == 1, "TFLite does not support multi-axis yet"
            axis = int(axis)

        return relax.op.flip(input_expr, axis)

    def get_expr(self, input_tensor_idx):
        return self._name2expr[_get_tensor_name(self.subgraph, input_tensor_idx)]

    def has_expr(self, input_tensor_idx):
        return _get_tensor_name(self.subgraph, input_tensor_idx) in self._name2expr

    def _new_const_or_var(self, value, dtype=None):
        """Return the Relax expr for tensor."""
        dtype = value.dtype if dtype is None else dtype
        if self._keep_params_in_input:
            var_name = f"_param_{len(self._params)}"
            new_var = relax.Var(var_name, relax.TensorStructInfo(value.shape, dtype))
            self._params[new_var] = tvm.nd.array(value)
            return new_var

        return relax.const(value, dtype)

    def get_tensor_expr(self, tensor, dtype=None):
        """Return the Relax expr for tensor."""
        if self.has_expr(tensor.tensor_idx):
            return self.get_expr(tensor.tensor_idx)

        value = self.get_tensor_value(tensor)
        dtype = value.dtype if dtype is None else dtype
        return self._new_const_or_var(value, dtype)

    def get_tensor_shape(self, tensor_wrapper):
        """Returns tensor shape. Infers shape if the shape is empty."""
        assert isinstance(tensor_wrapper, TensorWrapper), "Expecting TensorWrapper here"
        return (
            tensor_wrapper.tensor.ShapeAsNumpy()
            if tensor_wrapper.tensor.ShapeLength() > 0
            else self.get_tensor_expr(tensor_wrapper).struct_info.shape.values
        )


def _get_scalar_from_constant(expr):
    """Returns scalar value from Relax constant scalar."""
    is_constant_scalar = isinstance(expr, relax.Constant) and not expr.data.shape
    assert is_constant_scalar, "Expr is not a constant scalar."
    value = expr.data.numpy()
    msg = "value must be float32/int32"
    assert value.dtype in (np.dtype(np.int32), np.dtype(np.float32)), msg
    return value.item(0)


def _get_tensor_from_constant(expr):
    """Returns tensor of values from Relax constant node."""
    assert isinstance(expr, relax.Constant)
    value = expr.data.numpy()
    msg = "value must be float32/int32"
    assert value.dtype in (np.dtype(np.int32), np.dtype(np.float32)), msg
    return value


def _build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret


# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def _get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def _get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    tensor_name = subgraph.Tensors(tensor_idx).Name()
    if tensor_name is not None:
        tensor_name = tensor_name.decode("utf-8")
    else:
        tensor_name = "tvmgen_tensor_" + str(tensor_idx)
    return tensor_name


def _get_input_dict(model):
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    dtype_map = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        inputs_count = subgraph.InputsLength()
        assert inputs_count >= 1
        for input_index in range(inputs_count):
            input_ = subgraph.Inputs(input_index)
            assert subgraph.TensorsLength() > input_
            tensor = subgraph.Tensors(input_)
            input_shape = tuple(tensor.ShapeAsNumpy())
            tensor_type = tensor.Type()
            input_name = _get_tensor_name(subgraph, input_)
            shape_dict[input_name] = input_shape
            dtype_dict[input_name] = dtype_map[tensor_type]

    return shape_dict, dtype_dict


def from_tflite(model, shape_dict=None, keep_params_in_input=False):
    """Convert from tflite model into compatible relax Function.

    Parameters
    ----------
    model:
        tflite.Model or tflite.Model.Model (depending on tflite version)

    shape_dict : dict of str to int list/tuple
        Input shapes of the model.

    keep_params_in_input : bool
        If True, parameters will be treated as input variables. If false,
        parameters are treated as constant and folded directly into the graph.

    Returns
    -------
    mod : tvm.IRModule
        The relax module for compilation.
    """

    g = TFLiteGraphImporter(model, keep_params_in_input)
    return g.from_tflite(shape_dict)
