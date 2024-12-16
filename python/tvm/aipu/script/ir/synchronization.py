# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The synchronization part of IR APIs."""
from tvm import tir
from ..pysim import PyEvent, PySimInfo, random_pause
from .base import register_ir_api


@register_ir_api
def barrier():
    """All work-items in a work-group executing the kernel on a processor must execute this function
    before any are allowed to continue execution beyond the barrier. The barrier function will queue
    a memory fence to ensure correct ordering of memory operations to private memory and wait for
    events to finish between TECs.

    Examples
    --------
    .. code-block:: python

        S.barrier()

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: barrier
    """
    return tir.call_extern("void", "barrier", tir.precodegen("CLK_LOCAL_MEM_FENCE"))


@register_ir_api
def _py_barrier():
    py_sim_info = PySimInfo.current
    assert py_sim_info.is_multi_thread, "PySim can't be run in single thread for this DSL program."
    # Align with AIPU, barrier will wait all used events.
    wait_events(*tuple(x for x in py_sim_info.thread_local_data.events if not x.is_free))
    py_sim_info.barrier.wait()

    random_pause()


@register_ir_api
def alloc_events(count):
    """Allocates the specified number of events. The total number of events that the whole DSL
    program can allocate up to the concrete Zhouyi NPU target.

    Parameters
    ----------
    count : int
        The number of events you want to allocate for this invocation.

    Returns
    -------
    ret : Union[PrimExpr, Tuple[PrimExpr]]
        The events that already allocated.

    Examples
    --------
    .. code-block:: python

        ev0, ev1, ev2, ev3 = S.alloc_events(4)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_dma`
    """
    msg = 'The arg "count" expect an positive integer scalar constant.'
    assert isinstance(count, int) and count > 0, msg
    ret = (tir.call_extern("int32", "alloc_event"),) * count
    return ret[0] if count == 1 else ret


@register_ir_api
def _py_alloc_events(count):
    ret = tuple(x for x in PySimInfo.current.thread_local_data.events if x.is_free)[:count]
    assert count == len(ret), f"Only remain {len(ret)} events, but allocate {count}."

    for event in ret:
        event.is_free = False
    return ret[0] if count == 1 else ret


@register_ir_api
def wait_events(*events):
    """Blocks the current DSL program until all specified events are occurred.

    Parameters
    ----------
    events : Tuple[PrimExpr]
        The events need to be waited.

    Examples
    --------
    .. code-block:: python

        S.wait_events(ev0)
        S.wait_events(ev0, ev1, ev2)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_dma`
    """
    assert len(events) > 0, "Redundant useless call."
    bit_mask = 1 << events[0]
    for event in events[1:]:
        bit_mask |= 1 << event
    return tir.call_extern("void", "wait_events", bit_mask)


@register_ir_api
def _py_wait_events(*events):
    msg = 'The arg "events" expect objects returned by "S.alloc_events".'
    for event in events:
        assert isinstance(event, PyEvent), msg
        assert not event.is_free, "The released event can not be used anymore."
        event.wait()


@register_ir_api
def free_events(*events):
    """Release the allocated events, will call ``S.wait_events`` first automatically.

    Parameters
    ----------
    events : Tuple[PrimExpr]
        The events need to be released.

    Examples
    --------
    .. code-block:: python

        S.free_events(ev0)
        S.free_events(ev0, ev1, ev2)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_dma`
    """
    assert len(events) > 0, "Redundant useless call."
    bit_mask = 1 << events[0]
    for event in events[1:]:
        bit_mask |= 1 << event
    return tir.call_extern("void", "free_events", bit_mask)


@register_ir_api
def _py_free_events(*events):
    msg = 'The arg "events" expect objects returned by "S.alloc_events".'
    for event in events:
        assert isinstance(event, PyEvent), msg
        event.wait()
        event.reset()


__all__ = (
    "barrier",
    "alloc_events",
    "wait_events",
    "free_events",
)
