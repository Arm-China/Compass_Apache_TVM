# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Implement build relevant APIs of Zhouyi Compass extension of TIR."""
import os
import uuid
import tvm
from tvm import tir
from ..compass_info import CompassInfo
from ..utils import control_option, canonicalize_target
from ..logger import set_logger, DEBUG_ONCE, INFO
from . import _ffi_api, transform as compass_transform
from .analysis import extract_prim_func_info, has_perf_record_tick
from .cc import compile_c
from .executor import Executor
from .script.parser import parse_to_module, PyPrimFunc
from .utils import abspath


class CompassCodeGenCError(RuntimeError):
    """Error raised when a Compass OpenCL source code can't be generated successfully."""


class BuildManager:
    """The user interface of DSL program compilation.

    Examples
    --------
    .. code-block:: python

        @S.prim_func
        def add_func(xxx):
            xxx

        bm = BuildManager(target="X2_1204")
        mod = bm.lower(add_func)
        ex = bm.build(add_func)
    """

    def __init__(self, target="X2_1204", output_dir=None, cc_options="", disabled_pass=None):
        """Constructor.

        Parameters
        ----------
        target : Union[str, tvm.target.Target]
            The target that it is built for. It can be a literal target string or a
            tvm.target.Target object.

        output_dir : Optional[str]
            The directory to store all files generated during DSL program compilation and execution.
            If not set, a temporary directory inside the current working directory will be used.

        cc_options : Optional[str]
            The extra compilation options that need to be passed to the Compass OpenCL compiler.

        disabled_pass : Optional[Union[List[str], Tuple[str]]]
            The passes need to be disabled during DSL program compilation.
        """

        self._target = canonicalize_target(target)
        self.cps_info = CompassInfo.get(self._target)
        msg = f'The Compass DSL does not support the target "{self.cps_info.name}".'
        assert self.cps_info.version != "X1", msg

        self._output_dir = abspath(output_dir)
        self._cc_options = cc_options
        self._disabled_pass = disabled_pass
        self._is_from_schedule = True
        self.prim_func_info = None
        self._has_perf_tick = False

        self._cc_options += " -O0 -g" if control_option.is_debug else ""

        set_logger()
        DEBUG_ONCE(f"TVM {tvm.__version__} ({os.path.dirname(tvm.__file__)})")
        try:
            import op_lib  # pylint: disable=import-outside-toplevel

            DEBUG_ONCE(f"DSL OP Library Path: {os.path.dirname(op_lib.__file__)}")
        except ModuleNotFoundError:
            pass

    def _parse(self, inp, name):
        if isinstance(inp, PyPrimFunc):
            mod = parse_to_module(inp, name)
            self._is_from_schedule = False
        elif isinstance(inp, tvm.IRModule):
            mod = inp
        elif isinstance(inp, tir.PrimFunc):
            prim_func = inp.with_attr("global_symbol", name) if name else inp
            mod = tvm.IRModule({prim_func.attrs["global_symbol"]: prim_func})
        else:
            raise ValueError(f"Expected input to be an IRModule or PrimFunc, but got {type(inp)}.")

        if len(mod.functions) == 1:
            gv_name = list(mod.functions.keys())[0]
            prim_func = list(mod.functions.values())[0]
            prim_func = prim_func.with_attr("tir.is_entry_func", True)
            mod.update_func(gv_name, prim_func)

        return mod

    def lower(self, inp, name=None):
        """The lower interface of DSL program compilation.

        Parameters
        ----------
        inp : Union[tvm.tir.PrimFunc, tvm.IRModule]
            The TensorIR PrimFunc/IRModule to be lowered.

        name : Optional[str]
            The name of the result entry TensorIR PrimFunc. It is optional for TensorIR PrimFunc,
            and unused for TensorIR IRModule. For TensorIR PrimFunc, if set, it will override the
            current value of attribute "global_symbol" of the entry TensorIR PrimFunc.

        Returns
        -------
        ret : tvm.IRModule
            The result IRModule.
        """
        with self._target:
            mod = self._parse(inp, name)

        # PHASE 0
        passes = [
            compass_transform.SubstituteSizeVar(),
            compass_transform.AlignParamVarWithBuffer(),
            compass_transform.ReassignVarBy0DimBuffer(),
            compass_transform.SimplifyBufferIndex(),
        ]

        # PHASE 1
        passes += [
            compass_transform.BufRealizeSimplifier(),
            compass_transform.GenBufferStride(),
            tir.transform.LowerCrossThreadReduction(),
            tir.transform.LowerInitBlock(),
            tir.transform.Simplify(),
            tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tir.transform.ConvertBlocksToOpaque(),
            tir.transform.UnifyThreadBinding(),
            tir.transform.ManifestSharedMemoryLocalStage(),
        ]

        if self._is_from_schedule:
            passes += [
                tir.transform.CompactBufferAllocation(),
                compass_transform.MergeForWhere(),
            ]

        passes += [
            tir.transform.LowerMatchBuffer(),
            tir.transform.InjectSoftwarePipeline(),
            tir.transform.LowerOpaqueBlock(),
            tir.transform.FlattenBuffer(),
            tir.transform.NarrowDataType(32),
            tir.transform.Simplify(),
        ]

        if self._is_from_schedule:
            passes += [
                compass_transform.AddLikelyForLoopPartition(),
                tir.transform.LoopPartition(),
            ]

        passes += [
            tir.transform.Simplify(),
            tir.transform.HoistIfThenElse(),
            compass_transform.RemoveIfInVecFor(),
        ]

        if self._is_from_schedule:
            passes += [
                compass_transform.AddDMAPragma(),
            ]

        # PHASE 2
        passes += [
            tir.transform.RemoveNoOp(),
            tir.transform.VectorizeLoop(),
            tir.transform.InjectVirtualThread(),
            tir.transform.InjectDoubleBuffer(),
            tir.transform.StorageRewrite(),
            tir.transform.UnrollLoop(),
        ]

        # PHASE 3
        passes += [
            tir.transform.RenormalizeSplitPattern(),
            tir.transform.Simplify(),
            tir.transform.RemoveNoOp(),
            tir.transform.RewriteUnsafeSelect(),
            tir.transform.HoistIfThenElse(),
            compass_transform.RenameForLoopVar(),
        ]

        config = {
            "tir.LoopPartition": {"partition_const_loop": True},
            "tir.HoistIfThenElse": {"support_block_scope_hosting": True},
            "tir.DisablePointerValueTypeRewrite": True,
            "tir.Simplify": {
                "disable_var_inline": True,
                "convert_float_div_with_imm_to_mul": False,
            },
        }
        with self._target, tvm.transform.PassContext(
            opt_level=3, disabled_pass=self._disabled_pass, config=config
        ):
            return tvm.transform.Sequential(passes)(mod)

    def _codegen_c(self, ir_mod):
        passes = [
            tir.transform.BindTarget(self._target),
            compass_transform.InjectDma(self._target),
            compass_transform.CanonicalizeRamp(),
            compass_transform.RenameUtilsFnames(),
            compass_transform.RenameConstBufferVar(),
            compass_transform.InitializeEventState(),
            compass_transform.EliminateGetLocalID(),
            compass_transform.CanonicalizeDivMod(),
            tir.transform.CommonSubexprElimTIR(),
            compass_transform.ReassignVarByLet(),
            compass_transform.LowerStandard(),
            compass_transform.FoldConstant(),
            compass_transform.ExchangeConstantToRight(),
            compass_transform.Simplify(),
            compass_transform.LowerVirtualVectorPointer(),
            compass_transform.LowerTag(self.cps_info),
            compass_transform.HandleSubFuncReturnFWV(),
            compass_transform.AlignVectorWidthBySplit(self.cps_info),
            compass_transform.AlignVectorWidthByPad(self.cps_info),
            # After this line width of all vector nodes must equal to the hardware vector width.
            compass_transform.CombineInstructions(),
            compass_transform.ReviseParamR(),
            compass_transform.LowerVirtualIsa(),
            compass_transform.IsaAwareRewrite(),
            compass_transform.LowerVectorCast(self.cps_info),
            compass_transform.CombineInstructions(),
            compass_transform.ReviseParamR(),
            compass_transform.LowerPred(),
            # The passes after this line must not create "tir.const_pred" or "tir.low_true_pred"
            # nodes, so be careful when using script APIs in them.
            compass_transform.Revert2Standard(),
            compass_transform.UniquifyVarName(),
            compass_transform.Precodegen(),
        ]

        try:
            config = {"tir.Simplify": {"disable_var_inline": True}}
            with self._target, tvm.transform.PassContext(
                opt_level=3, disabled_pass=self._disabled_pass, config=config
            ):
                ir_mod = tvm.transform.Sequential(passes)(ir_mod)

            ret = _ffi_api.CodeGen(ir_mod, self.cps_info)
        except Exception as exc:
            raise CompassCodeGenCError(exc).with_traceback(exc.__traceback__) from None

        return ret

    def gen_op_lib(self, inp, name=None, output_path=None, verbose=False):
        """Generate Compass OpenCL code, compile it, and save it to the specified path.

        Parameters
        ----------
        inp : Union[tvm.tir.PrimFunc, tvm.IRModule]
            The TensorIR PrimFunc/IRModule to be built.

        name : Optional[str]
            The name of the result entry TensorIR PrimFunc. It is optional for TensorIR PrimFunc,
            and unused for TensorIR IRModule. For TensorIR PrimFunc, if set, it will override the
            current value of attribute "global_symbol" of the entry TensorIR PrimFunc.

        output_path : Optional[str]
            The path of the output object file that is compiled from the generated Compass OpenCL
            code file. If not set, it will be constructed using argument "output_dir" of the
            constructor and the name of entry PrimFunc.

        verbose : Optional[bool]
            Print output path if verbose is True. By default, it is False.
        """
        # 1. Lower to TIR IRModule.
        ir_mod = self.lower(inp, name)

        # 2. Get information of primitive function through analyzing the TIR IRModule.
        self.prim_func_info = extract_prim_func_info(ir_mod)
        self._has_perf_tick = has_perf_record_tick(ir_mod)
        func_name = self.prim_func_info.name
        if self._output_dir is None:
            self._output_dir = abspath(f"compass_dsl_{func_name}_{uuid.uuid4().hex}")
        if verbose:
            INFO(f"Current Output Path: {self._output_dir}")

        # 3. Generate Compass OpenCL code, compile and save them to disk.
        out_path = output_path or f"{self._output_dir}/gbuilder/op_lib/{func_name}.o"
        c_code = self._codegen_c(ir_mod)
        compile_c(c_code, self._target, self._cc_options, out_path, os.path.dirname(out_path))

    def build(self, inp, name=None):
        """The build interface of DSL program compilation.

        Parameters
        ----------
        inp : Union[tvm.tir.PrimFunc, tvm.IRModule]
            The TensorIR PrimFunc/IRModule to be built.

        name : Optional[str]
            The name of the result entry TensorIR PrimFunc. It is optional for TensorIR PrimFunc,
            and unused for TensorIR IRModule. For TensorIR PrimFunc, if set, it will override the
            current value of attribute "global_symbol" of the entry TensorIR PrimFunc.

        Returns
        -------
        ret : tvm.compass.tir.executor.Executor
            The object that is responsible for the subsequent execution job.
        """
        # 1. Generate Compass OpenCL code and compile it.
        self.gen_op_lib(inp, name, verbose=True)
        if isinstance(inp, PyPrimFunc):
            inp.param_infos = self.prim_func_info.param_infos
            inp.cps_info = self.cps_info
            inp.output_dir = self._output_dir

        # 2. Create "Executor" to responsible for the follow-up actions.
        return Executor(self.prim_func_info, self._output_dir, self.cps_info, self._has_perf_tick)
