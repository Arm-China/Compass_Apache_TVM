<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023 Arm Technology (China) Co. Ltd.-->

# Frequently Questioned Answers

**1. Some operators isn't partitioned to NPU subgraph because of OP Spec check,
how to skip the checkers?**

The option `disable_op_spec_checker` can be used to do that, set it in the
configuration file like below.
```
[Common]
disable_op_spec_checker = true
```
In addition, the corresponding override environment variable
`AIPU_TVM_DISABLE_OP_SPEC_CHECKER` can be used too.

**2. How to set the compute threshold for the pass `PruneAIPUSubGraphs`?**

The pass is used to prune low compute-intensive aipu subgraphs by compute threshold.

The option `compute_threshold` can be used to set the value, set it in the
configuration file like below.
```
[Common]
compute_threshold = 3e7
```
In addition, the corresponding override environment variable
`AIPU_TVM_COMPUTE_THRESHOLD` can be used too.

**3. How to see the graph snapshoots during the partition stage, e.g., the graph
before the pass `PruneAIPUSubGraphs`?**

The option `dump_partitioning_graph` can be used to do that, set it in the
configuration file like below.
```
[Common]
dump_partitioning_graph = true
```
In addition, the corresponding override environment variable
`AIPU_TVM_DUMP_PARTITIONING_GRAPH` can be used too.

The snapshoots will be dumped to the path
"***XXX***/partitioning_graph***Y***.txt" (***`XXX`*** is the root output
directory, ***`Y`*** is the internal partitioning stage number) as an unloadable
Relay IR, so you can use any editor to see which operators are partitioned to
NPU.

Currently there only is 1 partitioning stage, in another words,
`partitioning_graph0.txt` is the graph snapshoot before the pass
`PruneAIPUSubGraphs`.

**4. How to specify nodes to cpu during the partition stage?**
1. Set option `dump_annotation_graph` or environment variable
   `AIPU_TVM_DUMP_ANNOTATION_GRAPH` just like above.
2. Run first time.
3. Get node index that you want to specify to cpu from snapshoots "***XXX***/partitioning_annotation_graph.txt".
   (***`XXX`*** is the root output directory)
4. Run again with
    ```
    ...
    deployable = compass.compile(fallback_indices=[id1, id2, ...])
    ...
    ```
