<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use Inline Assembly

An inline assembly is a feature of the DSL in the Zhouyi NPU, which allows you to insert assembly instructions in the DSL source code. The compiler can distinguish these inline assembly snippet codes and treat them as low-level codes.

The inline assembly can be used in three situations:
- **Optimization**: You can use inline assembly code to implement the most performance-sensitive parts of the algorithms of your program. Machine code may be more efficient than what the compiler generates.
- **Zhouyi NPU specific instructions**: Some specific instructions in the Zhouyi NPU are not supported in the DSL. All these instructions can be accessed with inline assembly by DSL programmers.
- **Special functions**: Such as special calling conventions, hardware interrupts, and special directives for the linker and assembler.

You should be familiar with the assembly language of the Zhouyi NPU. For more information about the Zhouyi assembly language, see the *Arm China Zhouyi Compass AIPUv3 Assembly Programming Guide*.

```{warning}
**[PySim](../explanation/pysim.md) does not support inline assembly.** If your DSL program has used inline assembly, you cannot debug it directly in Python, only can run and debug it through the Compass simulator or hardware.
```


## Basic Syntax

The function of the inline assembly is `S.asm`.

The standard inline assembly should be used to identify instruction operands with variables. It allows you to set the input operands list, output operands list and clobbers list after the assembler template.

The function signature is as follows:

```python
def asm(
    template,
    outputs=None,
    inputs=None,
    clobbers=None,
    qualifiers=None
)
```

There are five parts of parameters to construct the main function of inline assembly code.


## Parameters

### Assembly template

This is a literal string that is the template for the assembler code. It is a combination of fixed text and tokens that refer to the input and output parameters. The string can contain any instructions recognized by the assembler, including directives.

The compiler does not parse the assembler instructions itself and does not know what they mean or even whether they are valid assembler inputs.

Operands with the specific register or immediate are identified in the same way as in assembly language (without any inline assembler prefix symbol).

With the assembly template, you can use:
- Semicolons (;) that are used in assembly code to separate multiple assembler instructions in a single inline assembly string.
- A newline character (\n) to break a line.
- A tab character (\t) to move to the instruction field for the pretty assembly code format.

You can also write bundled instructions because they comply with the normal syntax of assembly code. The single instruction is also supported.

For more information about the assembly instruction syntax, see the *Arm China Zhouyi Compass AIPUv3 Assembly Programming Guide*.

The following are some special format strings that are supported:
- **%%**: Outputs a single % into the assembler code.
- **%=**: Outputs a number that is unique to each instance of inline assembly code in the entire compilation. This option is useful when creating local labels and referring to them multiple times in a single template that generates multiple assembler instructions.

## Outputs

The outputs parameter is a dictionary which follows:

            {`asmSymbolicName`: [`constraint`, `VariableName`]}

The key `asmSymbolicName` is used to specify a symbolic name for the operand that is modified by the instructions in the assembler template.

The value is a list that contains two elements:

- The first element `constraint` is a string constant that specifies constraints of the operand. See the following constraints for details.
- The second element `VariableName` specifies a DSL variable that is mapping to `asmSymbolicName`. It must be writable.

An empty output dictionary is permitted.

A constraint is a string full of letters. The following are the basic letters that are allowed:

- **r**: This register operand is allowed if it is in a scalar register.
- **t**: This register operand is allowed if it is in a tensor register.
- **f**: This register operand is allowed if it is in a float register.
- **p**: This register operand is allowed if it is in a predicate register.

Multiple basic letters can be combined as constraint operands. These constraints are represented as multiple alternatives and the compiler can choose the best one as constraint operands.

Some modifier characters can be used to modify the constraint letters. The following are the modifiers that are allowed:
- **=**: This operand is written by this instruction.
- **+**: This operand is both read and written by the instruction. Others without = and + can only be read.
- **&**: This operand is an early-clobbered operand, which is written before the instruction finishes using the input operands. Only written operands can use &.

The following is a simple example for these cases:

```python
a = S.i32x8(0)
S.asm(
    "{add %[var_a].b, t1.b, t1.b;}",
    outputs={"var_a": ["=&t", a]}
)
```

## Inputs

The inputs parameter is a dictionary which follows:

            {`asmSymbolicName`: [`constraint`, `Expression`]}

- `asmSymbolicName` and `constraint` are the same as the ones in the outputs.

- `Expression` specifies a DSL variable or expression that is passed to the inline assembly as an input.

An empty input dictionary is permitted.

## Clobbers

This is a list of registers or other values that are changed by the assembler template, beyond those listed in the outputs section.

An empty clobber list is permitted.

## Qualifiers

- **volatile**: The typical use of extended asm statements is to manipulate input values to produce output values. However, your asm statements may also produce side effects. If so, you may need to use the volatile qualifier to disable certain optimizations.
- **inline**: If you use an inline qualifier, the assembler will take the inline assembly code as the possible smallest size.


## Example

The following is a simple example:

```python
byter1 = 0
byter2 = 1
byter3 = 2
S.asm(
    "add %[namex], %[namey], %[namez];",
    outputs={"namex": ["=&r", byter1]},
    inputs={"namey": ["r", byter2], "namez": ["r", byter3]}
)
```

This example will produce the following OpenCL code:

```c++
int byter1 = 0;
int byter2 = 1;
int byter3 = 2;
__asm__(
    "add %[namex], %[namey], %[namez];"
    : [namex] "=&r"(byter1)
    : [namey] "r"(byter2), [namez] "r"(byter3)
);
```
