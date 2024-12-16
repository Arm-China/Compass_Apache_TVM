<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Statement

A statement is a group of expressions and/or statements that you design to carry out a task or an
action. Statements are two-sided - that is, they either perform tasks or do not perform tasks.

## Return
Return is used to end the execution of the function call and "returns" the result (value of the
expression following the return keyword) to the caller.

```py
@S.prim_func
def add_one(x: S.int32) -> S.int32:
    return x + 1

@S.prim_func
def test_stmt(a: S.int32, out: S.int32):
    out = add_one(a)
```

## Assignment
Assignment is used to assign the value to the specified variable.

```py
@S.prim_func
def test_stmt(a: S.int32, out: S.int32):
    b, d = a + 1, a
    out = b + d
```

In this case, there are 3 variables assigned values.

## Allocate

Allocate is used to allocate memory or resources, similar to the "new" statement in C++, requires
three arguments: shape, dtype and scope.

- shape. The shape of data in the content of the buffer.
- dtype. The data type in the content of the buffer.
- scope. The storage scope of data in the content of the buffer.

```py
    @S.prim_func
    def test_stmt(a: S.int32, out: S.int32):
        lsram = S.alloc_buffer((1024,), "int32", scope="lsram")
        S.dma_memset(lsram, value=1, num=1024)
        out = lsram[0] + lsram[1]
```

In this case, we allocate a buffer on local SRAM with shape (1024,) and data type int32.

## Buffer Match

The buffer match function, matches a buffer from an address with the given shape.

- pointer. The address of a buffer.
- shape. The specific shape of the buffer that you want to match.

```py
@S.prim_func
def func(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
    a = S.match_buffer(A, shape=(h, w))
    b = S.match_buffer(B, shape=(w, h))
    for i, j in S.grid(h, w):
        b[j, i] = a[i, j]
```

In this case, the A and B are pointers of the address. We use S.match_buffer to match 2-D buffer a and b.
Then we can access data by subscript.

## For

```py
@S.prim_func
def test_stmt(a: S.int32, out: S.int32):
    out = 0
    for idx in range(a):
        out += idx
```

In this case, “for” is a statement defining the iteration step. In addition, “while” is also a form of loop statement.

## IfThenElse

In this case, "if" is a conditional statement.

```py
@S.prim_func
def test_stmt(a: S.int32, out: S.int32):
    if a == 1:
        out = 10
    else:
        out = 20
```

It decides the direction (control flow) of the flow of program execution.
