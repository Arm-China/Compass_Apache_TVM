# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Extract cell state and hidden state of lstm to tuple."""
from tvm import relay


class CellStateAndHiddenStateToTupleOutputExtractor(relay.ExprMutator):
    """Extract cell state and hidden state of lstm to tuple."""

    def __init__(self):
        super().__init__()
        self.tuple_out_map = dict()

    def visit_call(self, call):
        ret = super().visit_call(call)

        def is_match_lstm(qnn_sigmoid_after_split):
            # Match qnn.add->split->qnn.sigmoid
            if not isinstance(qnn_sigmoid_after_split.args[0], relay.TupleGetItem):
                return False
            if not isinstance(
                qnn_sigmoid_after_split.args[0].tuple_value, relay.Call
            ) or qnn_sigmoid_after_split.args[0].tuple_value.op != relay.op.get("split"):
                return False
            if not isinstance(
                qnn_sigmoid_after_split.args[0].tuple_value.args[0], relay.Call
            ) or qnn_sigmoid_after_split.args[0].tuple_value.args[0].op != relay.op.get("qnn.add"):
                return False
            # Match qnn.add+constant->qadd_before_split
            qadd_before_split = qnn_sigmoid_after_split.args[0].tuple_value.args[0]
            if not isinstance(qadd_before_split.args[0], relay.Call) or qadd_before_split.args[
                0
            ].op != relay.op.get("qnn.add"):
                return False
            if not isinstance(qadd_before_split.args[1], relay.Constant):
                return False
            # Match qnn.requantize0+qnn.requantize1->qadd_after_fc
            qadd_after_fc = qadd_before_split.args[0]
            if not isinstance(qadd_after_fc.args[0], relay.Call) or not isinstance(
                qadd_after_fc.args[1], relay.Call
            ):
                return False
            if qadd_after_fc.args[0].op != relay.op.get("qnn.requantize") or qadd_after_fc.args[
                1
            ].op != relay.op.get("qnn.requantize"):
                return False
            # Match qnn.dense->qnn.requantize0, qnn.dense->qnn.requantize1
            qreq_after_dense0, qreq_after_dense1 = qadd_after_fc.args[0], qadd_after_fc.args[1]
            if not isinstance(qreq_after_dense0.args[0], relay.Call) or not isinstance(
                qreq_after_dense1.args[0], relay.Call
            ):
                return False
            if qreq_after_dense0.args[0].op != relay.op.get("qnn.dense") or qreq_after_dense1.args[
                0
            ].op != relay.op.get("qnn.dense"):
                return False
            return True

        if ret.op == relay.op.get("qnn.mul"):
            qnn_mul = ret
            in_0, in_1 = qnn_mul.args[:2]
            if not isinstance(in_0, relay.Call) or not isinstance(in_1, relay.Call):
                return qnn_mul
            # Match qnn.sigmoid+(qnn.add->qnn.tanh)->qnn.mul
            if in_0.op == relay.op.get("qnn.sigmoid") and in_1.op == relay.op.get("qnn.tanh"):
                if not isinstance(in_1.args[0], relay.Call) or in_1.args[0].op != relay.op.get(
                    "qnn.add"
                ):
                    return qnn_mul
                qnn_add = in_1.args[0]
                qnn_sigmoid = in_0
                if not is_match_lstm(qnn_sigmoid):
                    return qnn_mul

                # Replace qnn.mul with tuple(qnn.mul, qnn.add)->tuplegetitem0
                out_exprs = [qnn_mul, qnn_add]
                out_tuple = relay.Tuple(out_exprs)
                qnn_mul_out = relay.TupleGetItem(out_tuple, 0)
                qnn_add_out = relay.TupleGetItem(out_tuple, 1)
                self.tuple_out_map.update({qnn_mul: qnn_mul_out})
                self.tuple_out_map.update({qnn_add: qnn_add_out})
                return qnn_mul_out
            # Match qnn.sigmoid+qnn.add->qnn.mul
            elif in_0.op == relay.op.get("qnn.sigmoid") and in_1.op == relay.op.get("qnn.add"):
                # Match qnn.mul+((qnn.sigmoid+qnn.tanh)->qnn.mul)->qnn.add
                qnn_add = in_1
                qmul_before_qadd0, qmul_before_qadd1 = qnn_add.args[:2]
                if not isinstance(qmul_before_qadd0, relay.Call) or not isinstance(
                    qmul_before_qadd1, relay.Call
                ):
                    return qnn_mul
                if qmul_before_qadd0.op != relay.op.get(
                    "qnn.mul"
                ) or qmul_before_qadd1.op != relay.op.get("qnn.mul"):
                    return qnn_mul
                qsigmoid_before_qmul, qtanh_before_qmul = qmul_before_qadd1.args[:2]
                if not isinstance(qsigmoid_before_qmul, relay.Call) or not isinstance(
                    qtanh_before_qmul, relay.Call
                ):
                    return qnn_mul
                if qsigmoid_before_qmul.op != relay.op.get(
                    "qnn.sigmoid"
                ) or qtanh_before_qmul.op != relay.op.get("qnn.tanh"):
                    return qnn_mul
                if not is_match_lstm(qsigmoid_before_qmul):
                    return qnn_mul
                if qnn_add not in self.tuple_out_map:
                    return qnn_mul

                # Replace qnn.add->qnn.mul with tuple(qnn.mul, qnn.add)->tuplegetitem1->qnn.mul
                new_args = [qnn_mul.args[0], self.tuple_out_map[qnn_add], *qnn_mul.args[2:8]]
                return relay.Call(
                    qnn_mul.op, new_args, qnn_mul.attrs, qnn_mul.type_args, qnn_mul.span
                )
            else:
                return qnn_mul
        return ret


@relay.transform.function_pass(opt_level=0)
class ExtractCellStateAndHiddenStateToTupleOutput:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return CellStateAndHiddenStateToTupleOutputExtractor().visit(func)
