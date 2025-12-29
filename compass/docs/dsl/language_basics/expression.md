<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Expression

An expression is any word or group of words or symbols, which is a value. In DSL, an expression is a value, or anything that executes and ends up being a value.

In DSL, there are 4 types of expressions.

## Var

```py
S.prim_func
def test_expr(a : S.int32):
    return a
```

In this case, “a” is a var, and it is an expression.

## Algebraic Operations

```py
S.prim_func
def test_expr(a : S.int32):
    return (a + 12) * (a - 1)
```

In this case, the return value is algebraic operations between expressions.


## Function Call

```py
S.prim_expr
def add_one(a : S.int32):
    return a + 1


S.prim_func
def test_expr(a: S.int32)
    return add_one(a) * (add_one(a) + 1)
```

In this case, the function call is an expression. Buildin call also belongs to expression.

## Buffer Load

```py
S.prim_func
def test_expr(a: S.ptr("int32"))
    return a[0];
```

In this case, the buffer load is expression.
