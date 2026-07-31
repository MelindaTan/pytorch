import itertools

import torch
from torch._C._distributed_c10d import HookOpName


_hook_id_counter = itertools.count(0x4E414E43)  # 'NANC'


_OP_NAME_MAP = {
    HookOpName.SEND: "send",
    HookOpName.RECV: "recv",
    HookOpName.BROADCAST: "broadcast",
    HookOpName.ALLREDUCE: "allreduce",
    HookOpName.REDUCE: "reduce",
    HookOpName.ALLGATHER: "allgather",
    HookOpName.REDUCE_SCATTER: "reduce_scatter",
    HookOpName.ALLTOALL: "alltoall",
    HookOpName.BARRIER: "barrier",
}


def _check_tensors(tensors, label, op_name):
    for i, t in enumerate(tensors):
        if t.is_floating_point():
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise RuntimeError(
                    f"NaN/Inf detected in {label} tensor[{i}] for '{op_name}'"
                )


class NanCheckHook:
    """Hook that checks for NaN values in tensors before collective operations.

    Registers a pre-hook on a ProcessGroup that inspects input and/or output
    tensors for NaN/Inf values. If detected, raises a RuntimeError before the
    collective runs, preventing corruption from propagating across ranks.

    Works with any backend (NCCL, Gloo, etc.).
    """

    def __init__(self, check_inputs=True, check_outputs=False):
        self._check_inputs = check_inputs
        self._check_outputs = check_outputs
        self._hook_id = None
        self._pg = None

    def attach(self, pg):
        if self._pg is not None:
            raise RuntimeError("NanCheckHook is already attached to a ProcessGroup")

        self._hook_id = next(_hook_id_counter)
        self._pg = pg

        def _pre_hook(args):
            op_name = _OP_NAME_MAP.get(args.name, "unknown")
            if self._check_inputs:
                _check_tensors(args.input_tensors, "input", op_name)
            if self._check_outputs:
                _check_tensors(args.output_tensors, "output", op_name)

        pg.register_pre_hook(self._hook_id, _pre_hook)
        return self

    def remove(self):
        if self._pg is None:
            raise RuntimeError("NanCheckHook is not attached to a ProcessGroup")
        self._pg.unregister_pre_hook(self._hook_id)
        self._pg = None
        self._hook_id = None

    @property
    def is_attached(self):
        return self._pg is not None
