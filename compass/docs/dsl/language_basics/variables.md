<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Variables
## Literal
### Scalar Literal
#### Default Type
If you directly write the integer and float scalar literal, the default data type will be `S.int32` and `S.float32`.
```py
x = 1     # S.int32
y = 3.14  # S.float32
```
#### Explicit Type
You can also explicitly indicate the datatype:
```py
x = S.int8(100)
y = S.uint8(255)
z = S.float16(1.2)
```

### Vector Literal
Vector literals can be used to create a vector from a list of scalars.
1. Initialize with a single scalar

    In this case, the scalar value is replicated across all lanes of the vector.
    ```py
    va = S.int32x8(1)
    ```
2. Initialize with a list of scalars
    ```py
    vb = S.int32x8([1, 2, 3, 4, 5, 6, 7, 8])
    ```
   The total number of literals should be the same lane of the datatype. If not, here is an example:
    ```py
    vb = S.int32x8([1, 2, 3, 4])
    #error: too few elements in vector initialization (expected 8 elements, have 4)
    ```

    **Note**: The `expr` in S.dtype(expr) should be literal, if not, please move the data construction outside the primfunc:
    ```py
    data = list(range(32))
    @S.prim_func
    def func():
        xxx
        va = S.int8x32(data)
    ```


## Variable
### Variable Declaration
1. Variable declaration with init_value

    ```py
    x = 1        # int32
    y = 3.14     # float32

    x = S.int8(100)
    y = S.uint8(255)
    z = S.float16(1.2)
    ```
2. You can declare multiple variables in one line:
   ```py
   x, y = 1, 2
   ```


### Variable Reassignment (Update Value)
1. Duplicating assignments for one variable is not allowed.

   The left handside of assignment cannot have duplicated variables.
    ```py
    a,a = 1,2
    ```
    `error: Duplicate vars assigned.`
2. If you reassign a variable, the new value dtype should be the same as the origin dtype
    ```py
    x = 1
    x = 2   # ok, same dtype
    ```

    It will throw an error if you reassign with different dtype:
    ```py
    x = 1
    x = 3.14
    #error: Type mismatch assignment: "int32" vs. "float32", need to do the explicit type conversion for the right hand side(i.e., new value).
    ```
    You should perform explicit type conversion.
    ```py
    x = 1
    x = S.int32(3.14)   #ok, explicit type cast
    ```
### Variable Access/Lookup (Scope)

**What is the scope of a variable?**

The scope of a variable is its lifetime in the program. This means that the scope of a variable is the block of code in the entire program where the variable is declared, used, and can be modified.

`The Variable scope in Compass DSL is the same as that in C`.

In this section, you will learn how local variables work in C/Compass DSL.

#### Example 1

Here is the first example:
```C
int x = 10;
{
    x = x + 1;
}
```

In C, you delimit a block of code by {}. The inner block is able to access the value of x that is declared in the outter block, and modify it by adding 1 to it.

In Compass DSL, the if statement will create an inner block.
```py
x = S.int32(10)
if True:
    x = x + 1
```
The inner block can access the `x` and modify it. The final x will be 11.

#### Example 2

Here is another related example.
```c
{
    int x = 7;
}
print("x is %d",x);
```
In this program, `x` is initialized in the inner block, and we are trying to access and print the value of the inner block's `x` on the outter block. When you compile this code, you'll get the error `error: 'x' undeclared`. This is because the variable `x` is declared in the inner block and its scope is limited to the inner block. In other words, it is `local` to the inner block and cannot be accessed from the outter block.

In Compass DSL, the similar example is:
```py
if (cond):
    x = 10
else:
    x = 5
a = x + 1   # error: name 'x' is not defined
```
You'll get the error for the last line:  `error: name 'x' is not defined`.

Instead, you can declare x and initialize it at the beginning:
```py
x = 0       #init x in the outter block
if (cond):
    x = 10
else:
    x = 5
a = x + 1   # OK
```
#### Generic Principle for Local Scoping of Variables
Based on the above examples, the generic principle for local scoping of variables is:
```
{
    /*OUTER BLOCK*/
    {
        //contents of the outer block just before the start of this block
        //CAN be accessed here

        /*INNER BLOCK*/
    }
    //contents of the inner block are NOT accessible here
}
```
