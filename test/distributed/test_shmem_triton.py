# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.distributed._symmetric_memory as symm_mem

if not torch.backends.cuda.is_built() or not symm_mem.is_nvshmem_available():
    print("SHMEM backend (NVSHMEM/rocSHMEM) not available, skipping tests")
    sys.exit(0)

# Shared SHMEM Triton tests for both NVSHMEM (CUDA) and rocSHMEM (ROCm).

import triton.language as tl

import torch.distributed as dist
import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
from torch._inductor.runtime.triton_compat import triton
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.inductor_utils import IS_H100, requires_triton


shmem_backend = shmem_triton.get_shmem_backend_module()
requires_shmem = shmem_triton.requires_shmem

device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_shmem
@triton.jit
def my_put_kernel(dest, src, nelems, pe):
    shmem_backend.put(dest, src, nelems, pe)


@requires_shmem
@triton.jit
def my_get_kernel(dest, src, nelems, pe, nbi: tl.constexpr):
    if nbi:
        shmem_backend.get_nbi(dest, src, nelems, pe)
        shmem_backend.quiet()
    else:
        shmem_backend.get(dest, src, nelems, pe)


@requires_shmem
@triton.jit
def my_putmem_signal_block_kernel(
    dst,
    src,
    size_bytes,
    signal,
    sig_val,
    sig_op,
    peer,
):
    shmem_backend.putmem_signal_block(
        dst, src, size_bytes, signal, sig_val, sig_op, peer
    )


@requires_shmem
@triton.jit
def my_signal_wait_until_kernel(signal, cmp_op, cmp_val):
    shmem_backend.signal_wait_until(signal, cmp_op, cmp_val)


@requires_shmem
@triton.jit
def my_signal_op_kernel(
    sig_addr,
    signal,
    sig_op,
    peer,
):
    shmem_backend.signal_op(sig_addr, signal, sig_op, peer)


@requires_shmem
@triton.jit
def my_wait_until_kernel(ivar, cmp_op, cmp_val):
    shmem_backend.wait_until(ivar, cmp_op, cmp_val)


@requires_shmem
@triton.jit
def my_fence_kernel():
    shmem_backend.fence()


@requires_shmem
@triton.jit
def my_put_with_fence_kernel(
    dst1,
    src1,
    dst2,
    src2,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    shmem_backend.put(dst1, src1, nelems, peer)
    shmem_backend.fence()
    shmem_backend.put(dst2, src2, nelems, peer)
    shmem_backend.fence()
    shmem_backend.put(flag_dst, flag_src, 1, peer)


@requires_shmem
@triton.jit
def my_put_with_quiet_kernel(
    dst,
    src,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    shmem_backend.put(dst, src, nelems, peer)
    shmem_backend.quiet()
    shmem_backend.put(flag_dst, flag_src, 1, peer)


@requires_shmem
@triton.jit
def my_barrier_test_kernel(dst, src, nelems):
    # Validate device-side barrier by coordinating put/compute phases in one launch.
    my_pe = shmem_backend.my_pe()
    n_pes = shmem_backend.n_pes()

    if my_pe == 0:
        p_src = src.to(tl.pointer_type(tl.int32))
        tl.store(p_src, 42)
        i = 1
        while i < n_pes:
            shmem_backend.put(dst, src, nelems, i)
            i += 1

    shmem_backend.barrier_all()

    if my_pe != 0:
        p_dst = dst.to(tl.pointer_type(tl.int32))
        received = tl.load(p_dst)
        tl.store(p_dst, received + 1)


@requires_shmem
@triton.jit
def my_sync_test_kernel(local_data, remote_data, nelems):
    my_pe = shmem_backend.my_pe()
    n_pes = shmem_backend.n_pes()

    p_local = local_data.to(tl.pointer_type(tl.int32))
    unique_value = my_pe + 100
    tl.store(p_local, unique_value)

    # sync_all makes local writes visible before subsequent remote reads.
    shmem_backend.sync_all()

    next_pe = (my_pe + 1) % n_pes
    shmem_backend.get(remote_data, local_data, nelems, next_pe)


@requires_shmem
@triton.jit
def my_barrier_all_kernel():
    shmem_backend.barrier_all()


@requires_shmem
@triton.jit
def my_alltoall_kernel(
    team_handle,
    dst,
    src,
    nelems_per_pe,
):
    shmem_backend.alltoall(team_handle, dst, src, nelems_per_pe)


@requires_shmem
@triton.jit
def my_broadcast_kernel(
    team_handle,
    dst,
    src,
    nelems,
    pe_root,
):
    shmem_backend.broadcast(team_handle, dst, src, nelems, pe_root)


@requires_shmem
@triton.jit
def my_reduce_kernel(
    team_handle,
    dest_tensor,
    source_tensor,
    nreduce,
    operation: tl.constexpr,
):
    shmem_backend.reduce(team_handle, dest_tensor, source_tensor, nreduce, operation)


class ShmemTritonTestBase(MultiProcContinuousTest):
    # """
    # Abstract base class for SHMEM Triton tests.
    #
    # This class provides a unified base for SHMEM tests (NVSHMEM, rocSHMEM).
    # Backend-specific skip policy is expressed with explicit decorators.
    #
    # SHMEMTritonTest is the single concrete test class that is collected.
    #
    # For backend-specific tests:
    # - If few, add methods to SHMEMTritonTestBase with:
    #     @unittest.skipIf(torch.version.hip is not None, "NVSHMEM-only")
    #     @unittest.skipIf(torch.version.hip is None, "rocSHMEM-only")
    # - If many, split into NVSHMEMTritonTest/ROCSHMEMTritonTest subclasses with class-level skips.
    # Start with decorated methods here; refactor if backend-only tests grow.
    __test__ = False
    backend_name = "NVSHMEM"

    def setUp(self) -> None:
        super().setUp()
        if self.__class__ is ShmemTritonTestBase:
            self.skipTest("Abstract SHMEM base test class")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_device(self) -> None:
        device_module.set_device(self.device)
        symm_mem.set_backend(self.backend_name)

    @requires_triton()
    def test_triton_put(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        nelems = 5
        dtype = torch.int64
        val = 42 + rank

        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        for i in range(nelems):
            src[i] = val * 10 + i

        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        dist.barrier()

        peer = 1 - rank
        if rank == 0:
            my_put_kernel[(1,)](dst, src, nelems, peer)

        dist.barrier()
        if rank == 1:
            expected = [420 + i for i in range(nelems)]
            torch.testing.assert_close(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    def _run_triton_get(self, nbi: bool) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 8
        dtype = torch.int8
        val = 7

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            my_get_kernel[(1,)](out, inp, numel, peer, nbi=nbi)

        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    def _run_triton_get_ring(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        world_size = dist.get_world_size()
        numel = 8
        dtype = torch.int8

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = (rank - 1) % world_size
        my_get_kernel[(1,)](out, inp, numel, peer, nbi=False)

        expected_value = peer
        torch.testing.assert_close(
            out, expected_value * torch.ones(numel, dtype=dtype, device=self.device)
        )

    @requires_triton()
    @parametrize("nbi", [False, True])
    def test_triton_get(self, nbi: bool) -> None:
        self._run_triton_get(nbi=nbi)

    @requires_triton()
    def test_triton_get_ring(self) -> None:
        self._run_triton_get_ring()

    @requires_triton()
    def test_triton_wait_until(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0
        FLAG_INITIAL_VALUE = 0
        FLAG_FINAL_VALUE = 42

        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(
            FLAG_INITIAL_VALUE
        )
        symm_mem.rendezvous(flag, group=group_name)
        expected_flag = torch.tensor(
            [FLAG_FINAL_VALUE], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=FLAG_FINAL_VALUE,
            )
            torch.testing.assert_close(flag, expected_flag)

        if rank == 1:
            my_put_kernel[(1,)](flag, expected_flag, 1, peer)

    @requires_triton()
    def test_triton_signal_wait_until(self) -> None:
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0
        NVSHMEM_SIGNAL_SET = 0
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val_to_put = 123
        completion_flag_val = 1
        flag_dtype = torch.int64

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val_to_put)
        symm_mem.rendezvous(inp, group=group_name)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=flag_dtype).fill_(0)

        if rank == 0:
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=completion_flag_val,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )
        elif rank == 1:
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=completion_flag_val,
            )
            torch.testing.assert_close(
                out, val_to_put * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag,
                torch.tensor(
                    [completion_flag_val], dtype=flag_dtype, device=self.device
                ),
            )

    @requires_triton()
    def test_triton_fence(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        dtype = torch.int8
        numel = 8
        val1 = 10
        val2 = 20
        flag_val = 1
        NVSHMEM_CMP_EQ = 0

        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp1, group=group_name)
        symm_mem.rendezvous(inp2, group=group_name)
        symm_mem.rendezvous(out1, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        symm_mem.rendezvous(flag, group=group_name)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            my_put_with_fence_kernel[(1,)](
                out1,
                inp1,
                out2,
                inp2,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 1:
            my_wait_until_kernel[(1,)](flag, cmp_op=NVSHMEM_CMP_EQ, cmp_val=flag_val)
            torch.testing.assert_close(
                out1, val1 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                out2, val2 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([flag_val], dtype=torch.int32, device=self.device)
            )

    @requires_triton()
    def test_triton_quiet(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        dtype = torch.int8
        numel = 8
        val = 15
        flag_val = 42
        NVSHMEM_CMP_EQ = 0

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(flag, group=group_name)

        dist.barrier()
        if rank == 1:
            my_put_with_quiet_kernel[(1,)](
                out,
                inp,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 0:
            my_wait_until_kernel[(1,)](flag, cmp_op=NVSHMEM_CMP_EQ, cmp_val=flag_val)
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
        dist.barrier()

    @requires_triton()
    def test_triton_barrier(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 1
        dtype = torch.int32

        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        my_barrier_test_kernel[(1,)](
            dst,
            src,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )
        dist.barrier()

        if rank == 0:
            torch.testing.assert_close(
                src, torch.tensor([42], device=self.device, dtype=dtype)
            )
        else:
            torch.testing.assert_close(
                dst, torch.tensor([43], device=self.device, dtype=dtype)
            )

    @requires_triton()
    def test_triton_sync(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 1
        dtype = torch.int32

        local_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        remote_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(local_data, group=group_name)
        symm_mem.rendezvous(remote_data, group=group_name)

        my_sync_test_kernel[(1,)](
            local_data,
            remote_data,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )

        expected_local = rank + 100
        torch.testing.assert_close(
            local_data, torch.tensor([expected_local], device=self.device, dtype=dtype)
        )

        next_rank = (rank + 1) % self.world_size
        expected_remote = next_rank + 100
        torch.testing.assert_close(
            remote_data, torch.tensor([expected_remote], device=self.device, dtype=dtype)
        )

    @requires_triton()
    def test_triton_put_signal_set(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_SET = 0
        SIGNAL_VAL = 1
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )

        if rank == 1:
            my_signal_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "Known hang in rocSHMEM Triton put_signal_add path.",
    )
    def test_triton_put_signal_add(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_ADD = 5
        SIGNAL_VAL = 16
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_ADD,
                peer=peer,
            )

        if rank == 1:
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op.",
    )
    def test_triton_alltoall(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        rank = self.rank
        nelems_per_pe = 2
        dtype = torch.int64
        src_size = nelems_per_pe * world_size
        src = symm_mem.empty(src_size, dtype=dtype, device=self.device)
        for i in range(world_size):
            value = rank * 100 + i
            src[i * nelems_per_pe : (i + 1) * nelems_per_pe] = value
        dst = symm_mem.empty(src_size, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        dist.barrier()
        team_handle = 0
        my_alltoall_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems_per_pe,
            launch_cooperative_grid=True,
        )
        dist.barrier()
        for i in range(world_size):
            expected = i * 100 + rank
            actual = dst[i * nelems_per_pe : (i + 1) * nelems_per_pe]
            torch.testing.assert_close(actual, torch.full_like(actual, expected))

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op.",
    )
    def test_triton_broadcast(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        nelems = 4
        dtype = torch.int64
        pe_root = 0
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        if rank == pe_root:
            for i in range(nelems):
                src[i] = 100 + i
        else:
            src.fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        dist.barrier()
        team_handle = 0
        my_broadcast_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems,
            pe_root,
            launch_cooperative_grid=True,
        )
        dist.barrier()
        expected = [100 + i for i in range(nelems)]
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op.",
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ],
    )
    def test_triton_sum_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        nreduce = 3
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            src[i] = (self.rank + 1) * (i + 1)
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        expected = []
        for i in range(nreduce):
            total = sum((r + 1) * (i + 1) for r in range(world_size))
            expected.append(total)
        dist.barrier()
        team_handle = 0
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="sum",
            launch_cooperative_grid=True,
        )
        dist.barrier()
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op.",
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ],
    )
    def test_triton_minmax_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        nreduce = 2
        src_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        src_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            if i == 0:
                src_min[i] = 10 + self.rank * 5
                src_max[i] = 10 + self.rank * 5
            else:
                src_min[i] = 20 - self.rank * 15
                src_max[i] = 20 - self.rank * 15
        dst_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        dst_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src_min, group=group_name)
        symm_mem.rendezvous(src_max, group=group_name)
        symm_mem.rendezvous(dst_min, group=group_name)
        symm_mem.rendezvous(dst_max, group=group_name)
        all_values = []
        for i in range(nreduce):
            values = []
            for r in range(world_size):
                if i == 0:
                    values.append(10 + r * 5)
                else:
                    values.append(20 - r * 15)
            all_values.append(values)
        expected_min = [min(vals) for vals in all_values]
        expected_max = [max(vals) for vals in all_values]
        dist.barrier()
        team_handle = 0
        my_reduce_kernel[(1,)](
            team_handle,
            dst_min,
            src_min,
            nreduce,
            operation="min",
            launch_cooperative_grid=True,
        )
        my_reduce_kernel[(1,)](
            team_handle,
            dst_max,
            src_max,
            nreduce,
            operation="max",
            launch_cooperative_grid=True,
        )
        dist.barrier()
        torch.testing.assert_close(
            dst_min, torch.tensor(expected_min, device=self.device, dtype=dtype)
        )
        torch.testing.assert_close(
            dst_max, torch.tensor(expected_max, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op.",
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ],
    )
    def test_triton_prod_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        nreduce = 3
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            if i == 0:
                src[i] = 1 if self.rank % 2 == 0 else 2
            elif i == 1:
                src[i] = 1
            else:
                src[i] = 1 if (self.rank // 2) % 2 == 0 else 2
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        vals = torch.empty(nreduce, world_size, dtype=dtype)
        vals[0, ::2] = 1
        vals[0, 1::2] = 2
        vals[1] = 1
        for rank in range(world_size):
            vals[2, rank] = 1 if (rank // 2) % 2 == 0 else 2
        expected = vals.prod(-1).tolist()
        dist.barrier()
        team_handle = 0
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="prod",
            launch_cooperative_grid=True,
        )
        dist.barrier()
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )


instantiate_parametrized_tests(ShmemTritonTestBase)


class SHMEMTritonTest(ShmemTritonTestBase):
    __test__ = True

# class-level skip for NVSHMEM Triton tests on non-H100 platforms.
SHMEMTritonTest = skip_but_pass_in_sandcastle_if(
    torch.version.hip is None and not IS_H100,
    "NVSHMEM Triton tests require H100.",
)(SHMEMTritonTest)


if __name__ == "__main__":
    run_tests()
