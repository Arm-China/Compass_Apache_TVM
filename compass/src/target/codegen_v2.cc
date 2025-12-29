// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/target/codegen_v2.cc
 */
#include "codegen_v2.h"

#include <compass/tvm/utils.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/name_supply.h>

#include <map>
#include <set>
#include <vector>

#include "../../../../src/arith/pattern_match.h"
#include "../../../../src/runtime/thread_storage_scope.h"
#include "../../../../src/support/str_escape.h"
#include "../../../../src/target/source/codegen_params.h"

namespace tvm {
namespace codegen {

CodeGenCompassV2::CodeGenCompassV2(CompassInfo cps_info) : cps_info_(cps_info) {
  restrict_keyword_ = "restrict";
  decl_stream << "#include <compass/dsl.h>\n\n";
  return;
}

void CodeGenCompassV2::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  DeclareFunction(gvar, f);
  auto function_name = GetFunctionName(gvar);

  // clear previous generated state.
  InitFuncState(f);

  this->stream << "\n";
  PrintFunctionSignature(function_name, f, stream);
  this->stream << " {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  // sync between kernel
  if (is_entry_) {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCompassV2::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  is_entry_ = false;
  attrs_ = "";

  if (f->HasNonzeroAttr("tir.is_entry_func")) {
    is_entry_ = true;
    attrs_ += "__kernel ";
  }

  Optional<Bool> is_inline = f->GetAttr<Bool>("is_inline");
  if (is_inline != nullptr) {
    if (is_inline.value() == true) {
      attrs_ += "_CLC_INLINE ";
    } else {
      attrs_ += "_CLC_NOINLINE ";
    }
  }
}

void CodeGenCompassV2::PrintFuncPrefix(std::ostream& os) { os << attrs_; }

void CodeGenCompassV2::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else {
    os << "get_group_id(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = os.str();
}

std::string CodeGenCompassV2::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  if (!dma_macro_decl_stream_.str().empty()) {
    code << dma_macro_decl_stream_.str() << "\n";
  }
  code << fwd_decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenCompassV2::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t.is_bool()) {
    os << "bool";
    if (t.is_vector_bool()) {
      os << lanes;
    }
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    os << "bf16";
    if (lanes == 1) return;
    os << lanes;
    return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 1:
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16 || lanes == 32)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Compass type";
}

std::string CodeGenCompassV2::PrintType(DataType t) {
  std::ostringstream os;
  PrintType(t, os);
  return os.str();
}

void CodeGenCompassV2::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintStorageScope(ptr->storage_scope, os);
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

void CodeGenCompassV2::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "" || StrStartsWith(scope, "global")) {
    // The scope of the cases that lowered from Relax is "".
    os << "__global ";
  } else if (scope == "shared") {
    os << "__local ";
  } else if (scope == "lsram") {
    os << "__lsram ";
  } else if (scope == "local") {
    return;
  } else if (scope == "constant") {
    os << "__constant ";
  } else {
    LOG(FATAL) << "Unsupported storage scope \"" << scope << "\".";
  }
}

void CodeGenCompassV2::VisitExpr_(const CastNode* op, std::ostream& os) {
  std::stringstream value;
  this->PrintExpr(op->value, value);
  if (op->value.dtype() == op->dtype) {
    os << value.str();
  } else {
    ICHECK_EQ(op->dtype.lanes(), 1) << "Only do scalar cast.";
    os << "(";
    this->PrintType(op->dtype, os);
    os << ")" << value.str();
  }
}

std::string CodeGenCompassV2::GetBufferRef(DataType dtype, const BufferNode* buffer,
                                           PrimExpr index) {
  DataType buf_dtype = buffer->dtype;
  const Var& buf_var = buffer->data;

  ICHECK_EQ(dtype, buf_dtype);  // Don't know the mismatch scenario, so error out now.
  // Don't know the volatile scenario, so error out now.
  ICHECK_EQ(IsVolatile(buf_var.get()), false);

  std::string buf_var_str = this->PrintExpr(buf_var);
  std::ostringstream oss;

  auto pointer_type = Downcast<PointerType>(buf_var->type_annotation);
  DataType buf_var_dtype = Downcast<PrimType>(pointer_type->element_type)->dtype;
  if (buf_dtype != buf_var_dtype) {
    oss << "((";
    this->PrintStorageScope(pointer_type->storage_scope, oss);
    this->PrintType(buf_dtype, oss);
    oss << "*)" << buf_var_str << ")";
  } else {
    oss << buf_var_str;
  }
  oss << "[" << this->PrintExpr(index) << "]";

  return oss.str();
}

void CodeGenCompassV2::PrintCallExtern(Type ret_type, String global_symbol,
                                       const Array<PrimExpr>& args, bool skip_first_arg,
                                       std::ostream& os) {
  os << global_symbol << "(";
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

void CodeGenCompassV2::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1) << "CodeGenCompassV2 only supports flat memory allocations.";
    os << "((";
    auto pointer_type = Downcast<PointerType>(load->buffer->data->type_annotation);
    PrintStorageScope(pointer_type->storage_scope, os);
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer->data.get()) << " + ";
    this->PrintExpr(load->indices[0], os);
    os << ')';
  } else if (op->op.same_as(builtin::reinterpret())) {
    auto target_dtype = op->dtype;
    os << "as_";
    this->PrintType(target_dtype, os);
    os << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(builtin::reassign())) {
    ICHECK_EQ(op->args.size(), 2U);
    auto var = Downcast<Var>(op->args[0]);
    PrimExpr value = op->args[1];

    this->PrintExpr(var, os);
    os << " = ";
    this->PrintExpr(value, os);
  } else if (op->op.same_as(builtin::pointer())) {
    ICHECK_EQ(op->args.size(), 4U);
    DataType dtype = op->args[0].dtype();
    std::string scope = Downcast<StringImm>(op->args[1])->value;
    PrimExpr base = op->args[2];
    PrimExpr offset = op->args[3];

    DataType base_dtype;
    if (const auto* var = base.as<VarNode>()) {
      if (!var->dtype.is_handle()) {
        // Indicate this pointer is created by getting address of a variable.
        base_dtype = var->dtype;
      } else {
        auto pointer_type = Downcast<PointerType>(var->type_annotation);
        base_dtype = Downcast<PrimType>(pointer_type->element_type)->dtype;
      }
    } else {
      auto call = Downcast<Call>(base);
      if (call->op.same_as(builtin::pointer())) {
        base_dtype = call->args[0]->dtype;
      } else if (call->op.same_as(builtin::reinterpret())) {
        base_dtype = call->dtype;
        base = call->args[0];
      } else {
        ICHECK(call->op->IsInstance<GlobalVarNode>());
        // The parser will guarantee the pointer that returned by this sub function call has the
        // same type of the current pointer.
        base_dtype = dtype;
      }
    }

    if (is_zero(offset) == false) os << "(";

    if (base_dtype != dtype) {
      os << "(";
      PrintStorageScope(scope, os);
      PrintType(dtype, os);
      os << "*)";
    }

    if (base->IsInstance<VarNode>() && !base->dtype.is_handle() && !base_dtype.is_handle()) {
      os << "&";  // Indicate this pointer is created by getting address of a variable.
    }
    this->PrintExpr(base, os);

    if (is_zero(offset) == false) {
      os << " + ";
      this->PrintExpr(offset, os);
      os << ")";
    }
  } else if (op->op.same_as(Op::Get("tir.vector_get_element"))) {
    ICHECK_EQ(op->args.size(), 2U);
    PrimExpr expr = op->args[0];
    PrimExpr idx = op->args[1];

    this->PrintExpr(expr, os);
    os << "[";
    this->PrintExpr(idx, os);
    os << "]";
  } else if (op->op.same_as(Op::Get("tir.vector_set_element"))) {
    ICHECK_EQ(op->args.size(), 3U);
    PrimExpr var_or_buffer_load = op->args[0];
    PrimExpr idx = op->args[1];
    PrimExpr value = op->args[2];

    this->PrintExpr(var_or_buffer_load, os);
    os << "[";
    this->PrintExpr(idx, os);
    os << "] = ";
    this->PrintExpr(value, os);
  } else if (op->op.same_as(builtin::call_extern())) {
    std::string name = Downcast<StringImm>(op->args[0])->value;
    const std::map<std::string, std::vector<std::string>> compare_names{
        {"vceq", {"__vqeq", "__vceq"}}, {"vcneq", {"__vqne", "__vcneq"}},
        {"vcge", {"__vqge", "__vcge"}}, {"vcgt", {"__vqgt", "__vcgt"}},
        {"vcle", {"__vqle", "__vcle"}}, {"vclt", {"__vqlt", "__vclt"}},
    };
    if (name == "vload") {
      ICHECK_EQ(op->args.size(), 4U);
      std::string ptr_str = this->PrintExpr(op->args[1]);
      PrimExpr stride = op->args[2];
      std::string mask_str = this->PrintExpr(op->args[3]);

      if (stride->IsInstance<IntImmNode>() && Downcast<IntImm>(stride)->value == 1) {
        os << "__vload(" << ptr_str << ", " << mask_str << ")";
        return;
      }

      std::string stride_str = this->PrintExpr(stride * op->dtype.bytes());
      os << "__vload_stride(" << ptr_str << ", " << stride_str << ", " << mask_str << ")";
      return;
    } else if (name == "vstore") {
      ICHECK_EQ(op->args.size(), 5U);
      std::string value_str = this->PrintExpr(op->args[1]);
      std::string ptr_str = this->PrintExpr(op->args[2]);
      PrimExpr stride = op->args[3];
      std::string mask_str = this->PrintExpr(op->args[4]);

      if (stride->IsInstance<IntImmNode>() && Downcast<IntImm>(stride)->value == 1) {
        os << "__vstore(" << value_str << ", " << ptr_str << ", " << mask_str << ")";
        return;
      }

      std::string stride_str = this->PrintExpr(stride * op->args[1]->dtype.bytes());
      os << "__vstore_stride(" << value_str << ", " << ptr_str << ", " << stride_str << ", "
         << mask_str << ")";
      return;
    } else if (name == "vload_gather") {
      size_t arg_cnt = op->args.size();
      ICHECK(4U <= arg_cnt && arg_cnt <= 5U);
      std::string ptr_str = this->PrintExpr(op->args[1]);
      std::string offsets0_str = this->PrintExpr(op->args[2]);
      PrimExpr mask = op->args[arg_cnt - 1];

      os << "__vload_gather(" << ptr_str << ", " << offsets0_str;
      if (arg_cnt == 5U) {
        os << ", " << this->PrintExpr(op->args[3]);
      }
      if (!is_all_true_pred(mask)) {
        os << ", " << this->PrintExpr(mask);
      }
      os << ")";
      return;
    } else if (name == "vstore_scatter") {
      size_t arg_cnt = op->args.size();
      ICHECK(5U <= arg_cnt && arg_cnt <= 6U);
      std::string value_str = this->PrintExpr(op->args[1]);
      std::string ptr_str = this->PrintExpr(op->args[2]);
      std::string offsets0_str = this->PrintExpr(op->args[3]);
      PrimExpr mask = op->args[arg_cnt - 1];

      os << "__vstore_scatter(" << value_str << "," << ptr_str << ", " << offsets0_str;
      if (arg_cnt == 6U) {
        os << ", " << this->PrintExpr(op->args[4]);
      }
      if (!is_all_true_pred(mask)) {
        os << ", " << this->PrintExpr(mask);
      }
      os << ")";
      return;
    } else if (compare_names.find(name) != compare_names.end()) {
      ICHECK_EQ(op->args.size(), 4U);
      std::vector<std::string> names = compare_names.at(name);
      std::string func_name = op->args[1]->dtype.is_floating() ? names[0] : names[1];
      std::string x_str = this->PrintExpr(op->args[1]);
      std::string y_str = this->PrintExpr(op->args[2]);
      std::string mask_str = this->PrintExpr(op->args[3]);

      os << func_name << "(" << x_str << ", " << y_str << ", " << mask_str << ")";
      return;
    } else if (name == "vxtl" || name == "vxth") {
      ICHECK_EQ(op->args.size(), 2U);
      std::string func_name = op->dtype.is_int() ? "__vs" : "__vu";
      if (name == "vxtl") {
        func_name += "xtl";
      } else {
        func_name += "xth";
      }
      std::string x_str = this->PrintExpr(op->args[1]);

      os << func_name << "(" << x_str << ")";
      return;
    } else if (name == "vtbl") {
      size_t arg_cnt = op->args.size();
      ICHECK(4U <= arg_cnt && arg_cnt <= 6U);

      // The "vperm" has better performance than "vtbl".
      std::string func_name = arg_cnt == 4 ? "__vperm" : "__vtbl";

      size_t index_arg_idx = arg_cnt - 1;
      os << func_name << "(";
      for (size_t i = 1; i < index_arg_idx; ++i) {
        os << this->PrintExpr(op->args[i]) << ", ";
      }
      os << this->PrintExpr(op->args[index_arg_idx]) << ")";
      return;
    } else if (name == "vadd" || name == "vsub") {
      ICHECK_EQ(op->args.size(), 6U);
      PrimExpr x = op->args[2];
      PrimExpr y = op->args[3];
      PrimExpr mask = op->args[4];
      bool with_saturate = Downcast<Bool>(op->args[5]);

      std::string x_str = this->PrintExpr(x);
      std::string y_str = this->PrintExpr(y);

      if (x->dtype == y->dtype && is_all_true_pred(mask) && with_saturate == false) {
        std::string symbol_name = name == "vadd" ? " + " : " - ";
        os << "(" << x_str << symbol_name << y_str << ")";
        return;
      }

      std::string func_name = "__" + name;
      func_name += with_saturate ? "s" : "";

      if (name == "vadd" && with_saturate && x->dtype.is_int() && y->dtype.is_uint()) {
        x_str = this->PrintExpr(y);
        y_str = this->PrintExpr(x);
      }

      std::string r_str = this->PrintExpr(op->args[1]);
      std::string mask_str = this->PrintExpr(mask);

      os << func_name << "(" << r_str << ", " << x_str << ", " << y_str << ", " << mask_str << ")";
      return;
    } else if (name == "vmul") {
      ICHECK_EQ(op->args.size(), 5U);
      PrimExpr x = op->args[2];
      PrimExpr y = op->args[3];
      PrimExpr mask = op->args[4];

      std::string x_str = this->PrintExpr(x);
      std::string y_str = this->PrintExpr(y);

      if (x->dtype == y->dtype && is_all_true_pred(mask)) {
        os << "(" << x_str << " * " << y_str << ")";
        return;
      }

      ICHECK(op->dtype.is_floating()) << "There isn't any integer equal-width multiply functions.";
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string mask_str = this->PrintExpr(mask);

      os << "__vmul(" << r_str << ", " << x_str << ", " << y_str << ", " << mask_str << ")";
      return;
    } else if (name == "vmull" || name == "vmulh") {
      ICHECK_EQ(op->args.size(), 5U);
      std::string func_name = name == "vmull" ? "__vmul" : "__vmulh";
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string x_str = this->PrintExpr(op->args[2]);
      std::string y_str = this->PrintExpr(op->args[3]);
      std::string mask_str = this->PrintExpr(op->args[4]);

      os << func_name << "(" << r_str << ", " << x_str << ", " << y_str << ", " << mask_str << ")";
      return;
    } else if (name == "vabs") {
      ICHECK_EQ(op->args.size(), 5U);
      std::string func_name = Downcast<Bool>(op->args[4]) ? "__vabss" : "__vabs";
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string x_str = this->PrintExpr(op->args[2]);
      std::string mask_str = this->PrintExpr(op->args[3]);

      os << func_name << "(" << r_str << ", " << x_str << ", " << mask_str << ")";
      return;
    } else if (name == "vnsr") {
      ICHECK_EQ(op->args.size(), 6U);
      PrimExpr x = op->args[1];
      bool with_saturate = Downcast<Bool>(op->args[4]);
      bool with_round = Downcast<Bool>(op->args[5]);

      std::string func_name = "__vnsr";
      func_name += x->dtype.is_int() ? "a" : "l";
      func_name += with_saturate ? "s" : "";
      func_name += with_round ? "r" : "";
      func_name += (op->dtype.bits() == 16) && (x->dtype.bits() == 32) ? "_h" : "";  // to_h
      if (with_saturate) {
        func_name += op->dtype.is_int() ? "_s" : "_u";
      }

      std::string x_str = this->PrintExpr(x);
      std::string shift_str = this->PrintExpr(op->args[2]);
      std::string mask_str = this->PrintExpr(op->args[3]);

      os << func_name << "(" << x_str << ", " << shift_str << ", " << mask_str << ")";
      return;
    } else if (name == "vsl" || name == "vsr") {
      ICHECK_EQ(op->args.size(), 6U);
      std::string func_name;
      if (name == "vsl") {
        bool with_saturate = Downcast<Bool>(op->args[5]);
        func_name = with_saturate ? "__vsasl" : "__vlsl";
      } else {  // vsr
        func_name = op->dtype.is_int() ? "__vasr" : "__vlsr";
        bool with_round = Downcast<Bool>(op->args[5]);
        func_name += with_round ? "r" : "";
      }

      std::string x_str = this->PrintExpr(op->args[2]);
      std::string shift_str = this->PrintExpr(op->args[3]);

      os << func_name << "(";
      PrimExpr mask = op->args[4];
      if (is_all_true_pred(mask)) {
        os << x_str << ", " << shift_str;
      } else {
        std::string r_str = this->PrintExpr(op->args[1]);
        std::string mask_str = this->PrintExpr(mask);
        os << r_str << ", " << x_str << ", " << shift_str << ", " << mask_str;
      }
      os << ")";
      return;
    } else if (name == "vror") {
      ICHECK_EQ(op->args.size(), 5U);
      std::string x_str = this->PrintExpr(op->args[2]);
      std::string y_str = this->PrintExpr(op->args[3]);

      os << "__vror(";
      PrimExpr mask = op->args[4];
      if (is_all_true_pred(mask)) {
        os << x_str << ", " << y_str;
      } else {
        std::string r_str = this->PrintExpr(op->args[1]);
        std::string mask_str = this->PrintExpr(mask);
        os << r_str << ", " << x_str << ", " << y_str << ", " << mask_str;
      }
      os << ")";
      return;
    } else if (name == "vand" || name == "vor" || name == "vxor") {
      ICHECK_EQ(op->args.size(), 5U);
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string x_str = this->PrintExpr(op->args[2]);
      std::string y_str = this->PrintExpr(op->args[3]);
      std::string mask_str = this->PrintExpr(op->args[4]);

      os << "__" << name << "(";
      if (op->dtype.is_bool() == false) {
        os << r_str << ", ";
      }
      os << x_str << ", " << y_str << ", " << mask_str << ")";
      return;
    } else if (name == "vinv") {
      ICHECK_EQ(op->args.size(), 4U);
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string x_str = this->PrintExpr(op->args[2]);
      std::string mask_str = this->PrintExpr(op->args[3]);

      os << "__vinv(";
      if (op->dtype.is_bool() == false) {
        os << r_str << ", ";
      }
      os << x_str << ", " << mask_str << ")";
      return;
    } else if (name == "__vbcast") {
      ICHECK_EQ(op->args.size(), 4U);
      std::string r_str = this->PrintExpr(op->args[1]);
      std::string value_str = this->PrintExpr(op->args[2]);
      PrimExpr mask = op->args[3];

      if (is_all_true_pred(mask)) {
        os << "(";
        PrintType(op->dtype, os);
        os << ")" << value_str;
        return;
      }
      os << "__vbcast(" << r_str << ", " << value_str << ", " << PrintExpr(mask) << ")";
      return;
    } else if (name == "vrevs") {
      ICHECK_EQ(op->args.size(), 2U);
      std::string func_name = op->dtype.is_bool() ? "__vprevs" : "__vrevs";

      os << func_name << "(" << this->PrintExpr(op->args[1]) << ")";
      return;
    } else if (name == "vzip") {
      ICHECK_EQ(op->args.size(), 4U);
      std::string x_str = this->PrintExpr(op->args[1]);
      std::string y_str = this->PrintExpr(op->args[2]);
      std::string part = Downcast<StringImm>(op->args[3])->value;

      os << "__vzip" << part[0] << "(" << x_str << ", " << y_str << ")";
      return;
    } else if (name == "vconcat") {
      std::string part = Downcast<StringImm>(op->args[op->args.size() - 1])->value;
      if (part == "all") {
        auto num_inps = op->args.size() - 2;
        os << "(";
        PrintType(op->dtype, os);
        os << ")(" << this->PrintExpr(op->args[1]);
        for (size_t i = 2; i <= num_inps; i++) {
          os << "," << this->PrintExpr(op->args[i]);
        }
        os << ")";
      } else {
        ICHECK_EQ(op->args.size(), 4U);
        std::string x_str = this->PrintExpr(op->args[1]);
        std::string y_str = this->PrintExpr(op->args[2]);

        os << "__vext" << part[0] << "(" << x_str << ", " << y_str << ")";
      }
      return;
    } else if (name == "inline_asm") {
      os << "__asm__" << this->PrintExpr(op->args[1]) << "(\n";  // Begin mark and qualifiers.
      int asm_scope = this->BeginScope();                        // Increase the indent.

      this->PrintIndent(os);
      os << this->PrintExpr(op->args[2]);  // The template consist of assembly code.

      // Print the output operands, input operands, and clobbers.
      for (size_t i = 3; i < op->args.size(); ++i) {
        if (const auto* colon_mark = op->args[i].as<StringImmNode>()) {
          if (colon_mark->value == ": ") {
            os << "\n";
            this->PrintIndent(os);
            os << ": ";
            continue;
          }
        }
        os << this->PrintExpr(op->args[i]);
      }
      os << "\n";

      this->EndScope(asm_scope);  // Decrease the indent.
      this->PrintIndent(os);
      os << ")";
      return;
    } else if (name == "AsyncDmaDirect" || name == "DmaDirect" || name == "DmaUpsample") {
      size_t arg_cnt = op->args.size();
      size_t arg_cnt_gt = 8U;
      if (name == "DmaDirect") {
        arg_cnt_gt = 7U;
      } else if (name == "DmaUpsample") {
        arg_cnt_gt = 10U;
      }
      ICHECK_EQ(arg_cnt, arg_cnt_gt);

      static const std::map<std::string, std::string> scope2dma_addr_base{
          {"global", "kGlobal"},    {"global.1", "kGlobal1"}, {"global.2", "kGlobal2"},
          {"global.3", "kGlobal3"}, {"lsram", "kLsram"},      {"shared", "kShared"},
          {"local", "kGlobal"}};

      // Generate function calling
      auto dst = Downcast<Call>(op->args[1]);
      auto src = Downcast<Call>(op->args[2]);
      std::string dst_scope = Downcast<StringImm>(dst->args[1])->value;
      std::string src_scope = Downcast<StringImm>(src->args[1])->value;
      std::string dst_base = scope2dma_addr_base.at(dst_scope);
      std::string src_base = scope2dma_addr_base.at(src_scope);

      std::string func_name = name + "_" + src_base + "_to_" + dst_base;
      os << func_name << "((int)" << this->PrintExpr(dst) << ", (int)" << this->PrintExpr(src);
      for (size_t i = 3; i < arg_cnt; ++i) {
        os << ", " << this->PrintExpr(op->args[i]);
      }
      os << ")";

      // Generate function code through Macros
      if (macro_instances_.find(func_name) != macro_instances_.end()) {
        return;
      }
      macro_instances_.insert(func_name);

      if (name == "AsyncDmaDirect") {
        dma_macro_decl_stream_ << "GEN_ASYNC_DMA_DIRECT_";
      } else if (name == "DmaDirect") {
        dma_macro_decl_stream_ << "GEN_DMA_DIRECT_";
      } else {
        dma_macro_decl_stream_ << "GEN_DMA_UPSAMPLE_";
      }
      if (StrStartsWith(src_scope, "global") && (dst_scope == "lsram" || dst_scope == "shared")) {
        dma_macro_decl_stream_ << "EXT2INT";
      } else {
        dma_macro_decl_stream_ << "INT2EXT";
      }
      dma_macro_decl_stream_ << "(" << src_base << ", " << dst_base << ");\n";
      return;
    } else if (name == "vall" || name == "vany") {
      ICHECK_EQ(op->args.size(), 2U);
      std::string func_name = (name == "vall") ? "__vandr" : "__vorr";
      std::string mask_str = this->PrintExpr(op->args[1]);

      os << func_name << "(" << mask_str << ")";
      return;
    } else {
      CodeGenC::VisitExpr_(op, os);
    }
  } else if (op->op.same_as(builtin::ret())) {
    if (op->args.size() == 0) {
      if (is_entry_) {
        os << "barrier(CLK_LOCAL_MEM_FENCE);";
      }
      os << "return";
    } else {
      os << "return ";
      this->PrintExpr(op->args[0], os);
    }
  } else if (op->op.same_as(builtin::Break())) {
    os << "break";
  } else if (op->op.same_as(builtin::if_then_else())) {
    std::string cond = PrintExpr(op->args[0]);
    std::string true_val = PrintExpr(op->args[1]);
    std::string false_val = PrintExpr(op->args[2]);
    os << "(" << cond << " ? " << true_val << " : " << false_val << ")";
  } else if (op->op.same_as(Op::Get("tir.precodegen"))) {
    ICHECK_EQ(op->args.size(), 1U);
    os << Downcast<StringImm>(op->args[0])->value;
  } else if (op->op.same_as(Op::Get("tir.vector_literal"))) {
    const Array<PrimExpr>& args = op->args;
    const size_t lanes = op->dtype.lanes();
    ICHECK_EQ(lanes, args.size());

    os << "(";
    PrintType(op->dtype, os);
    os << ")(";
    for (size_t i = 0; i < lanes; ++i) {
      this->PrintExpr(args[i], os);
      if (i < lanes - 1) {
        os << ",";
      }
    }
    os << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCompassV2::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "(";
  PrintType(op->dtype, os);
  os << ")";
  os << v;
}

void CodeGenCompassV2::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  auto dtype = op->dtype;
  ICHECK_LE(dtype.bits(), 32) << "CodeGenCompassV2 IntImmNode only supports types within 32 bits.";
  if (dtype == DataType::Int(32)) {
    if (op->value == -2147483648) {
      os << "(int)" << op->value;
    } else {
      os << op->value;
    }
  } else {
    os << "(";
    PrintType(dtype, os);
    os << ")" << op->value;
  }
}

void CodeGenCompassV2::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  auto dtype = op->dtype;
  ICHECK_LE(dtype.bits(), 32)
      << "CodeGenCompassV2 FloatImmNode only supports types within 32 bits.";
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      os << "-";
    }
    os << "INFINITY";
  } else if (std::isnan(op->value)) {
    os << "NAN";
  } else {
    if (dtype.bits() == 16) {
      os << "(";
      PrintType(dtype, os);
      os << ")";
    }

    // The value in hexadecimal floating-point format to preserve the exact binary value,
    // and additionally include the decimal form as a comment for readability.
    std::ostringstream temp;
    temp << std::hexfloat << op->value << 'f';
    temp << "/*" << std::scientific << op->value << "*/";
    os << temp.str();
  }
}

void CodeGenCompassV2::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "\"" << ::tvm::support::StrEscape(op->value) << "\"";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr, std::ostream& os, CodeGenCompassV2* p) {
  if (op->dtype.lanes() == 1) {
    std::string op_str(opstr);
    std::set<std::string> ops_needed_ret_type{"+", "-", "*", "/"};
    if (ops_needed_ret_type.count(op_str) && op->dtype != DataType::Int(32)) {
      os << "(" + p->PrintType(op->dtype) + ")";
    }

    if (isalpha(opstr[0])) {
      os << opstr << "(";
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenCompassV2::VisitExpr_(const AddNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "+", os, this);
}

void CodeGenCompassV2::VisitExpr_(const SubNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "-", os, this);
}

void CodeGenCompassV2::VisitExpr_(const MulNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "*", os, this);
}

void CodeGenCompassV2::VisitExpr_(const DivNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "/", os, this);
}

void CodeGenCompassV2::VisitExpr_(const MinNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "min", os, this);
}

void CodeGenCompassV2::VisitExpr_(const MaxNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "max", os, this);
}

void CodeGenCompassV2::VisitExpr_(const FloorDivNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "/", os, this);
}

void CodeGenCompassV2::VisitExpr_(const AndNode* op, std::ostream& os) {
  os << "(";
  this->PrintExpr(op->a, os);
  os << " && ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenCompassV2::VisitExpr_(const OrNode* op, std::ostream& os) {
  os << "(";
  this->PrintExpr(op->a, os);
  os << " || ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenCompassV2::VisitExpr_(const SelectNode* op, std::ostream& os) {
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeGenCompassV2::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  const DataType& dtype = op->dtype;
  ICHECK(dtype.is_scalar() || is_all_true_pred(op->predicate))
      << "Masked vector load should be handled by 'vload'.";

  os << GetBufferRef(dtype, op->buffer.get(), op->indices[0]);
  return;
}

void CodeGenCompassV2::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  std::string cond = PrintExpr(op->condition);
  stream << "while (" << cond << "){\n";
  int while_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenCompassV2::VisitStmt_(const BufferStoreNode* op) {
  const DataType& dtype = op->value->dtype;
  ICHECK(dtype.is_scalar() || is_all_true_pred(op->predicate))
      << "Masked vector store should be handled by 'vstore'.";

  std::string buffer_ref = GetBufferRef(dtype, op->buffer.get(), op->indices[0]);
  std::string value = PrintExpr(op->value);

  PrintIndent();
  stream << buffer_ref << " = " << value << ";\n";
  return;
}

void CodeGenCompassV2::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto pointer_type = Downcast<PointerType>(op->buffer_var->type_annotation);
  PrintStorageScope(pointer_type->storage_scope, stream);

  PrintType(op->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenCompassV2::VisitStmt_(const ForNode* op) {
  std::string min = PrintExpr(op->min);
  std::string max = PrintExpr(op->extent);
  auto annotations = op->annotations;
  std::string step = "1";
  if (annotations.count("step")) {
    step = PrintExpr(Downcast<PrimExpr>(annotations["step"]));
  }
  if (!is_zero(op->min)) {
    max = min + "+" + max;
  }

  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << min << "; " << vid << " < " << max << "; " << vid
         << " += " << step << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenCompassV2::VisitStmt_(const LetStmtNode* op) {
  // The value must be generated first, because some expressions may generate new
  // statement before this let statement, like what "tir.reinterpret" will do.
  std::string value_str = this->PrintExpr(op->value);

  this->PrintIndent();
  const VarNode* var = op->var.get();
  if (var->dtype.is_handle()) {
    auto pointer_type = Downcast<PointerType>(var->type_annotation);
    this->PrintType(pointer_type, this->stream);
  } else {
    this->PrintType(var->dtype, this->stream);
  }
  this->stream << " " << this->AllocVarID(var) << " = " << value_str << ";\n";

  this->PrintStmt(op->body);
  return;
}

void CodeGenCompassV2::VisitStmt_(const AllocateConstNode* op) {
  int64_t num_elements = 1;
  const runtime::NDArray& data = op->data.value();
  runtime::DataType data_type = data.DataType();
  for (int64_t dim : data.Shape()) {
    num_elements *= dim;
  }
  std::string symbol_name = AllocVarID(op->buffer_var.get());

  // Allocate the global static variable
  decl_stream << "\n__constant ";
  PrintType(data_type, decl_stream);
  decl_stream << " __attribute__((aligned(32))) " << symbol_name << "[" << num_elements
              << "] = {\n";

  NDArrayDataToC(data, 4, decl_stream);
  decl_stream << "};\n\n";
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
