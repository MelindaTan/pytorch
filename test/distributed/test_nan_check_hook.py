# Owner(s): ["oncall: distributed"]

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.nan_check_hook import NanCheckHook
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


NAN_HOOK_BACKENDS = [
    ("gloo", "cpu"),
    ("nccl", "cuda"),
]


class AbstractNanCheckHookTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.rank}")
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    def test_normal_tensors_pass(self):
        pg = self._init_pg()
        hook = NanCheckHook()
        hook.attach(pg)
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        hook.remove()

    def test_nan_input_raises(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=True)
        hook.attach(pg)
        t = torch.tensor([1.0, float("nan"), 3.0], device=self.device)
        with self.assertRaises(RuntimeError, msg="NaN detected"):
            dist.all_reduce(t)
        hook.remove()

    def test_inf_input_raises(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=True)
        hook.attach(pg)
        t = torch.tensor([1.0, float("inf"), 3.0], device=self.device)
        with self.assertRaises(RuntimeError, msg="NaN detected"):
            dist.all_reduce(t)
        hook.remove()

    def test_check_inputs_false_skips(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=False, check_outputs=False)
        hook.attach(pg)
        t = torch.tensor([1.0, float("nan"), 3.0], device=self.device)
        dist.all_reduce(t)
        hook.remove()

    def test_remove_stops_checking(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=True)
        hook.attach(pg)
        hook.remove()
        t = torch.tensor([1.0, float("nan"), 3.0], device=self.device)
        dist.all_reduce(t)

    def test_integer_tensors_skipped(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=True)
        hook.attach(pg)
        t = torch.tensor([1, 2, 3], device=self.device)
        dist.all_reduce(t)
        hook.remove()

    def test_multiple_ops(self):
        pg = self._init_pg()
        hook = NanCheckHook(check_inputs=True)
        hook.attach(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()
        t_nan = torch.tensor([float("nan")], device=self.device)
        with self.assertRaises(RuntimeError):
            dist.all_reduce(t_nan)
        hook.remove()


def _make_nan_hook_test_class(backend_name, device_type):
    class NanCheckHookTest(AbstractNanCheckHookTest, MultiProcessTestCase):
        pass

    NanCheckHookTest.backend_name = backend_name
    NanCheckHookTest.device_type = device_type
    NanCheckHookTest.__name__ = f"{backend_name.capitalize()}NanCheckHookTest"
    NanCheckHookTest.__qualname__ = NanCheckHookTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(NanCheckHookTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "NaN hook CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in NAN_HOOK_BACKENDS:
    globals()[f"{backend_name.capitalize()}NanCheckHookTest"] = (
        _make_nan_hook_test_class(backend_name, device_type)
    )


if __name__ == "__main__":
    run_tests()
