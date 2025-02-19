<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use DMA
This section describes how to use DMA in Compass DSL.

The DMA interface in Compass DSL includes two types: synchronous and asynchronous. We will first introduce the use of DMA in detail using the synchronous DMA interface, and then introduce the asynchronous DMA interface based on this.

## Synchronous DMA
### Prepare Buffer

Generally, DMA is used to move data between two different buffers. First we need to prepare buffer for DMA.

```py
@S.prim_func
def func(a: S.ptr(dtype, "global")):
    buffer_src = S.match_buffer(a, (100,))
    buffer_dst = S.alloc_buffer((100,), dtype, scope="lsram")
```

Here we define two buffers to move data from `buffer_src` to `buffer_dst`. Also, you can use ptr instead of buffer.

```py
ptr_src = S.ptr(dtype, scope="global")
ptr_dst = S.ptr(dtype, scope="lsram")
```

Here are some notes:
- The parameters buffer_src and buffer_dst must have the same dtype.
- Copying data between "lsram" and "shared", "lsram" and "lsram", "shared" and "shared" is not supported.

### Use DMA for Data Move

After preparing the buffer, we can directly move some data between these buffers.

```py
S.dma_copy(buffer_dst, buffer_src, width)
```

This means we move width data from buffer_src to buffer_dst. The width here is the number of data, not the number of bytes.

Of course you can also move data from buffer_dst to buffer_src, just swap their positions. Remember that the first buffer is the destination, and the second buffer is the source.

```py
S.dma_copy(buffer_src, buffer_dst, width)
```

### Use DMA with Stride

Sometimes we not only need to move data directly, but also need some strides between data. For example:

```
@@@@@xxx                @@@@@x
@@@@@xxx      ->        @@@@@x
@@@@@xxx                @@@@@x
@@@@@xxx                @@@@@x
```

Here we need to move `@`, but every 5 `@` will be separated by 3 `x`. Also in the destination, every 5 `@` will be separated by 1 `x`.

For source, everytime we move 5 data, and from the beginning to the end of this line, stride is 8. For destination, everytime we move 5 data, and from the beginning to the end of this line, stride is 6. This process is repeated 4 times in total.

So we can use DMA like:

```py
S.dma_copy(buffer_dst, buffer_src, width=5, src_stride=8, times=4, dst_stride=6)
```

This statement describes the above data transfer behavior. If stride is equal to width, there is no need to explicitly indicate its value.

## Asynchronous DMA

There is no difference between asynchronous DMA and synchronous DMA in the configuration of data transfer. The above content for synchronous DMA also applies to asynchronous DMA.

Compared with the synchronous DMA interface, the asynchronous DMA interface has an additional mandatory parameter: `event`.

You can think of an `event` as a synchronization state, which has two modes: idle and busy. When async DMA is bound to an event, it will first check whether the event is idle. If it is not idle, it will wait until it reaches the idle state before starting DMA and set the event to the busy state. After the DMA task is completed, the event will be reset to the idle state. At this time, the next async DMA task bound to this event will obtain this event and start its DMA task. In other words, multiple async DMA tasks that use the same event are serial. Of course, if multiple async DMA tasks are bound to different events, they can run in parallel in theory, but in practice this also depends on the specific hardware resources.

Events are obtained through the interface `S.alloc_events`. The input parameter is a number, indicating how many events to obtain (the total number of events that the whole DSL program can allocate is up to the specific Zhouyi NPU target).

For example, when two events are needed, call it like this:
```py
ev0, ev1 = S.alloc_events(2)
```

After getting the events, you can call the `S.async_dma_copy` interface to move the data. For example, the following is to move the data of `buffer_src_0` and `buffer_src_1` to `buffer_dst_0` and `buffer_dst_1` respectively:
```py
S.async_dma_copy(buffer_dst_0, buffer_src_0, width, event=ev0)
S.async_dma_copy(buffer_dst_1, buffer_src_1, width, event=ev1)
```

After starting the asynchronous DMA, you need to use the `S.wait_events` interface to wait for the specified asynchronous DMA task to complete. Generally speaking, this step is performed before you want to operate on the transferred data. For example, if you want to extract the data in `buffer_dst_0` for calculation, you must wait for the transfer from `buffer_src_0` to `buffer_dsl_0` to complete before performing the calculation. This can be done by calling `S.wait_events`.
```py
S.wati_events(ev0)
```

`S.wait_events` will block the current DSL program until all specified events occur, which means that you can also wait for the multiple events at the same time:

```py
S.wait_events(ev0, ev1)
```

Of course, those events you waited for must be allocated first.

After completing the transfer task, if you do not need to use the event anymore, you need to call `S.free_events` to release the event. For example, after completing the transfer from `buffer_src_0` to `buffer_dsl_0`, if you do not need to use `ev0` anymore, you can release `ev0` like this:
```py
S.free_events(ev0)
```

Or wait until all events are used and then release them together:
```py
S.free_events(ev0, ev1)
```

So in a nutshell, when using the asynchronous DMA interface to transfer data, you first need to get available events through `S.alloc_events`, then call `S.async_dma_copy`, in addition to configuring the parameters for data transfer, you also need to assign an event obtained in the previous step to it. Then, use `S.wait_events` to wait for the asynchronous DMA transfer to end at the appropriate location, and finally, after the asynchronous DMA ends, when the event is no longer needed, release the event through `S.free_events`.
