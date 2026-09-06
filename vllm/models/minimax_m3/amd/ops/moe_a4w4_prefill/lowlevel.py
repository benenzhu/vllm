# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-level FlyDSL helpers shared by the M3 a4w4 MoE kernels written in the
``ArithValue`` / raw-dialect style (sort.py, gemm1_1x4.py): raw global loads
and stores through an LLVM address-space-1 pointer, LDS atomics, buffer
resources from a byte pointer, torch tensor -> ``fx.Pointer`` arguments."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops  # the copy shipped in the vLLM image
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.expr import arith
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T


def ptr_arg(t: torch.Tensor):
    """torch tensor -> ``fx.Pointer`` argument (byte pointer to its storage)."""
    assert t.is_contiguous(), "ptr_arg needs a contiguous tensor"
    v = t.view(torch.uint8) if t.element_size() == 1 and t.dtype != torch.uint8 else t
    return flyc.from_c_void_p(fx.Uint8, v.data_ptr())


def ptr_buffer_resource(ptr, num_records_bytes):
    """Fat buffer resource (``!fly.ptr<i8, BufferDesc>``) over ``ptr``; use
    ``fx.rocdl.get_buffer_rsrc`` on it for raw ROCDL intrinsics."""
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64, num_records_bytes=num_records_bytes
    )


def raw_rsrc(fat):
    return fx.as_ir_value(fx.rocdl.get_buffer_rsrc(fat))


def extract_global_ptr(ptr):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), addr_i64)


def elem_offset_to_i64(elem_offset):
    if isinstance(elem_offset, int):
        return arith.constant(elem_offset, type=T.i64)
    raw = elem_offset.ir_value() if hasattr(elem_offset, "ir_value") else elem_offset
    if isinstance(raw.type, ir.IndexType):
        return arith.index_cast(T.i64, raw)
    int_type = ir.IntegerType(raw.type)
    if int_type.width == 64:
        return ArithValue(raw)
    return ArithValue(arith.ExtSIOp(T.i64, raw).result)


def _elem_ptr(global_ptr, elem_offset, elem_bytes):
    byte_offset_i64 = elem_offset_to_i64(elem_offset) * arith.constant(
        elem_bytes, type=T.i64
    )
    return buffer_ops.get_element_ptr(
        global_ptr, byte_offset=byte_offset_i64, elem_type=T.i8
    )


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def global_load_i32(global_ptr, elem_offset, *, nontemporal=False):
    return ArithValue(
        llvm.LoadOp(
            T.i32,
            _elem_ptr(global_ptr, elem_offset, 4),
            alignment=4,
            nontemporal=nontemporal,
        ).result
    )


def global_load_f32(global_ptr, elem_offset, *, nontemporal=False):
    return ArithValue(
        llvm.LoadOp(
            T.f32,
            _elem_ptr(global_ptr, elem_offset, 4),
            alignment=4,
            nontemporal=nontemporal,
        ).result
    )


def global_load_i32_vec(global_ptr, elem_offset, width, *, nontemporal=False):
    return llvm.LoadOp(
        T.vec(width, T.i32),
        _elem_ptr(global_ptr, elem_offset, 4),
        alignment=4,
        nontemporal=nontemporal,
    ).result


def global_store_i32(global_ptr, elem_offset, value, *, nontemporal=False):
    return llvm.StoreOp(
        _raw(value),
        _elem_ptr(global_ptr, elem_offset, 4),
        alignment=4,
        nontemporal=nontemporal,
    )


global_store_f32 = global_store_i32


def lds_i32_ptr(lds_memref, elem_offset):
    byte_offset_idx = ArithValue(
        elem_offset_to_i64(elem_offset) * arith.constant(4, type=T.i64)
    ).index_cast(T.index)
    base_idx = memref.extract_aligned_pointer_as_index(lds_memref)
    ptr_i64 = arith.index_cast(T.i64, base_idx + byte_offset_idx)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), ptr_i64)


def lds_atomic_add_i32(lds_memref, elem_offset, value):
    """Returns the old value."""
    return ArithValue(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            lds_i32_ptr(lds_memref, elem_offset),
            _raw(value),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def dpp_xor_f32(src, offset: int, *, bound_ctrl: bool = True):
    """f32 from lane ``L ^ offset`` (offset 1 or 2) via DPP quad permute."""
    src_i32 = (
        src.bitcast(T.i32)
        if hasattr(src, "bitcast")
        else ArithValue(src).bitcast(T.i32)
    )
    dpp_ctrl = {1: 0xB1, 2: 0x4E}[offset]
    out_i32 = llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.update.dpp.i32",
        [
            _raw(src_i32),
            _raw(src_i32),
            arith.constant(dpp_ctrl, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(bound_ctrl, type=ir.IntegerType.get_signless(1)),
        ],
        [],
        [],
    )
    return ArithValue(out_i32).bitcast(T.f32)
