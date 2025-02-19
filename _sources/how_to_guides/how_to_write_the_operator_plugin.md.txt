<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Write the Operator Plugin

```{note}
The plugin described in this article comes from the Compass buildtool. Here shows a simple demo based on
the pure Compass GBuilder mechanism. The DSL and TVM have no modification for the plugin. For
complete information, see section 7.4 of `<Arm China Zhouyi Compass NN Compiler User Guide>`.
```

The operator plugin is used to integrate operator implementation into the Compass buildtool. The Compass
buildtool matches the plugin through OpType, so one operator’s plugin can be implemented as multiple
small plugins, all these plugins are registered for the same OpType, and each plugin is responsible for a
specific situation. Certainly one operator’s plugin can be implemented as a single plugin as well.

This article will tell you:

* Python plugin rule
* How to implement a Python plugin
* How to test a plugin

## 1. Python Plugin Rule

The plugins should obey rules below:

- **Naming**: Use fixed **prefix** `aipubt_`, such as `aipubt_layernorm.py`.

## 2. Demo about Python Plugin

Create a file named `aipubt_layernorm.py`, first import required 3rd libraries and register OpType for LayerNorm:

```py
import os
from AIPUBuilder.core import BuilderOpPlugin, register_optype, BuilderParams, DataLayout
from AIPUBuilder.plugin_loader import register_plugin, PluginType
from AIPUBuilder.logger import DEBUG


op_type = register_optype("LayerNorm")
```

The main body of plugin code has 7 key functions.

### get_graph_pattern

Create a graph pattern based on nodes used by this plugin pattern.

```py
@register_plugin(PluginType.Builder, 0)
class LayerNormPlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        return ([("n1", op_type)], [])
```

### get_score

If several plugins have the same pattern, the Compass buildtool will choose the highest-score plugin.

```py
    def get_score(self):
        return 25
```

### check_params

```py
    def check_params(self, nodes):
        if self.target not in ["X2_1204", "X2_1204MP3"]:
            return False

        node = nodes[0]
        # check input output
        if str(node.inputs[0].dtype) != "float16" or str(node.outputs[0].dtype) != "float16":
            return False
        return True
```

### setup

Mainly used to define the input/output layouts of your operator.

```py
    def setup(self, sgnode, nodes):
        sgnode.attrs["keeping_layout"] = True
        valid_in = [[DataLayout.NHWC], [DataLayout.NDHWC], [DataLayout.Flat]]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True
```

### generate_code_name

Return your operator's binary object file name.

```py
    def generate_code_name(self, sgnode, nodes):
        DEBUG("DSL Plugin Picked (Python)")
        return "dsl_tpc_layernorm_fp16_x2.o"
```

### generate_params

Return an RO list. It corresponds to your parameters of the kernel entry function.


```py
    def generate_params(self, sgnode, nodes):
        ro = BuilderParams()
        in_tensor = sgnode.inputs[0]

        out_tensor = sgnode.outputs[0]
        axis_v = nodes[0].params["axis"]
        len_v = len(axis_v)
        dims = len(in_tensor.shape)

        shape0 = 1
        shape1 = 1
        for i in range(0, dims - len_v, 1):
            shape0 *= in_tensor.shape[i]

        for i in range(dims - len_v, dims, 1):
            shape1 *= in_tensor.shape[i]

        ro.append(shape0)
        ro.append(1)
        ro.append(1)
        ro.append(shape1)
        ro.append(in_tensor)
        ro.append(out_tensor)
        gamma = sgnode.constants[f"{nodes[0].name}/weights"]
        beta = sgnode.constants[f"{nodes[0].name}/biases"]
        ro.append(gamma)
        ro.append(beta)

        return ro
```

### generate_kernel

Attain the kernel entry function name and return.

```py
    def generate_kernel(self, sgnode, nodes):
        obj_name = os.path.basename(self.generate_code_name(sgnode, nodes))
        assert len(obj_name) > 2  # The name ends with '.o'
        kernel_name = obj_name[:-2]
        DEBUG(f"kernel_name:{kernel_name}")
        return kernel_name
```

## 3. Test Plugin

Here shows a demo using the `aipurun` (see section 4.2 of
`<Arm China Zhouyi Compass NN Compiler User Guide>` for details) command. Assume that you have completed the following:

- Set enviroment variable `AIPUPLUGIN_PATH` including plugin files you added.
- Operator object file: You need to build `layernorm` operator implementation to the object file in
  advance. Assume that you have prepared them in the `build` directory.
- IR files
  - `graph.def`: The graph structure part of the Compass IR.
  - `weight.bin`: The weight data part of the Compass IR.
  - `input0.bin`, `input1.bin`: The input binary files.

Run the following command:

```shell
export AIPUPLUGIN_PATH=${AIPUPLUGIN_PATH}:<directory-of-your-plugin-file>

aipurun graph.def \
  -i input0.bin,input1.bin \
  -w weight.bin \
  -L ./build/  # -L points to directory including operator object file

# Compare output.bin with gt.bin which you prepared first
diff gt.bin ./output.bin
```
