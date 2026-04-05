"""Triton kernels for TurboQuant KV cache quantization.

Provides fused encode (rotate + quantize) and decode (dequantize + unrotate) kernels.
Falls back to the PyTorch implementation in turboquant.py when Triton is unavailable.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    # Verify Triton has an active GPU driver (import succeeds even without GPU)
    try:
        triton.runtime.driver.active  # noqa: B018
        HAS_TRITON = True
    except (RuntimeError, AttributeError):
        HAS_TRITON = False
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _turboquant_encode_kernel(
        # Inputs
        input_ptr,
        R_ptr,
        outlier_mask_ptr,
        # Outputs
        codes_ptr,
        scale_ptr,
        outlier_ptr,
        # Strides
        stride_input_row,
        stride_R_row,
        stride_codes_row,
        stride_outlier_row,
        # Params
        D: tl.constexpr,  # head_dim
        n_normal: tl.constexpr,  # number of non-outlier channels
        n_outlier: tl.constexpr,  # number of outlier channels
        N_LEVELS: tl.constexpr,  # codebook size (e.g. 16)
        BLOCK_D: tl.constexpr,
    ):
        """Fused: rotation via tl.dot, per-vector scale, uniform quantize to uint8.

        Each program handles one (batch*head) row.
        Uses tl.dot for blocked matrix multiply instead of scalar loops.
        Quantization uses uniform rounding which approximates Lloyd-Max well
        post-rotation (Gaussian assumption holds after orthogonal transform).
        """
        row_idx = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        # Load x as [1, BLOCK_D] for tl.dot
        x = tl.load(
            input_ptr + row_idx * stride_input_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        x_2d = tl.reshape(x, [1, BLOCK_D])

        # Load full rotation matrix R [BLOCK_D, BLOCK_D]
        R_offs = offs[:, None] * stride_R_row + offs[None, :]
        R_mask = (offs[:, None] < D) & (offs[None, :] < D)
        R_block = tl.load(R_ptr + R_offs, mask=R_mask, other=0.0).to(tl.float32)

        # x_rot = x @ R via tl.dot — single blocked matmul
        x_rot_2d = tl.dot(x_2d, R_block)  # [1, BLOCK_D]
        x_rot = tl.reshape(x_rot_2d, [BLOCK_D])

        # Load outlier mask
        outlier_mask = tl.load(outlier_mask_ptr + offs, mask=mask, other=0).to(tl.int1)
        normal_mask = ~outlier_mask & mask

        # Compute absmax scale over normal channels only
        abs_normal = tl.where(normal_mask, tl.abs(x_rot), 0.0)
        scale = tl.max(abs_normal, axis=0)
        scale = tl.maximum(scale, 1e-8)
        tl.store(scale_ptr + row_idx, scale.to(tl.float16))

        # Uniform quantization: code = clamp(round((x_norm/scale + 1)/2 * (N-1)), 0, N-1)
        # This approximates Lloyd-Max well post-rotation since distribution is ~Gaussian
        normalized = x_rot / scale
        codes_float = (normalized + 1.0) * 0.5 * (N_LEVELS - 1)
        # libdevice.round is not always available; use floor(x + 0.5) for rounding
        codes_int = tl.math.floor(codes_float + 0.5).to(tl.int32)
        codes_int = tl.minimum(tl.maximum(codes_int, 0), N_LEVELS - 1)

        # Scatter normal channels into codes output using cumsum for compact indexing
        # normal_positions[j] = number of normal channels before j (exclusive prefix sum)
        normal_int = tl.where(normal_mask, 1, 0)
        normal_cumsum = tl.cumsum(normal_int, axis=0) - 1  # 0-based index

        # Store normal codes compactly
        tl.store(
            codes_ptr + row_idx * stride_codes_row + normal_cumsum,
            codes_int.to(tl.uint8),
            mask=normal_mask,
        )

        # Scatter outlier channels into outlier output
        outlier_int = tl.where(outlier_mask & mask, 1, 0)
        outlier_cumsum = tl.cumsum(outlier_int, axis=0) - 1

        tl.store(
            outlier_ptr + row_idx * stride_outlier_row + outlier_cumsum,
            x_rot.to(tl.float16),
            mask=outlier_mask & mask,
        )

    @triton.jit
    def _turboquant_decode_kernel(
        # Inputs
        codes_ptr,
        scale_ptr,
        outlier_ptr,
        outlier_mask_ptr,
        R_T_ptr,
        # QJL inputs (ignored if has_qjl is False)
        qjl_bits_ptr,
        qjl_norm_ptr,
        # Output
        output_ptr,
        # Strides
        stride_codes_row,
        stride_outlier_row,
        stride_RT_row,
        stride_output_row,
        stride_qjl_bits_row,
        # Params
        D: tl.constexpr,
        n_normal: tl.constexpr,
        n_outlier: tl.constexpr,
        N_LEVELS: tl.constexpr,
        has_qjl: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Dequantize + QJL correction + merge outliers + unrotate via tl.dot.

        Each program handles one (batch*head) row.
        """
        row_idx = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        # Load scale
        scale = tl.load(scale_ptr + row_idx).to(tl.float32)

        # Load outlier mask and build compact indices
        outlier_mask = tl.load(outlier_mask_ptr + offs, mask=mask, other=0).to(tl.int1)
        normal_mask = ~outlier_mask & mask

        normal_int = tl.where(normal_mask, 1, 0)
        normal_cumsum = tl.cumsum(normal_int, axis=0) - 1

        outlier_int = tl.where(outlier_mask & mask, 1, 0)
        outlier_cumsum = tl.cumsum(outlier_int, axis=0) - 1

        # Load and dequantize normal codes: uniform inverse
        # code -> value: val = (code / (N-1)) * 2 - 1 then * scale
        raw_codes = tl.load(
            codes_ptr + row_idx * stride_codes_row + normal_cumsum,
            mask=normal_mask,
            other=0,
        ).to(tl.float32)
        dequant_normal = (raw_codes / (N_LEVELS - 1) * 2.0 - 1.0) * scale

        # QJL residual correction on normal channels
        if has_qjl:
            # Load sign bits (uint8, one per normal channel)
            sign_raw = tl.load(
                qjl_bits_ptr + row_idx * stride_qjl_bits_row + normal_cumsum,
                mask=normal_mask,
                other=0,
            ).to(tl.float32)
            signs = sign_raw * 2.0 - 1.0  # {0,1} -> {-1,+1}

            # Load residual norm for this row
            res_norm = tl.load(qjl_norm_ptr + row_idx).to(tl.float32)

            # QJL correction: (||r|| * sqrt(pi/2) / n_normal) * sign
            # pi/2 ≈ 1.5707963
            correction = signs * (res_norm * 1.2533141 / n_normal)  # sqrt(pi/2) ≈ 1.2533141
            dequant_normal = dequant_normal + correction

        # Load outlier values
        outlier_vals = tl.load(
            outlier_ptr + row_idx * stride_outlier_row + outlier_cumsum,
            mask=outlier_mask & mask,
            other=0.0,
        ).to(tl.float32)

        # Merge into full rotated vector [BLOCK_D]
        rotated = tl.zeros([BLOCK_D], dtype=tl.float32)
        rotated = tl.where(normal_mask, dequant_normal, rotated)
        rotated = tl.where(outlier_mask & mask, outlier_vals, rotated)

        # Unrotate: output = rotated @ R_T via tl.dot
        rotated_2d = tl.reshape(rotated, [1, BLOCK_D])

        R_T_offs = offs[:, None] * stride_RT_row + offs[None, :]
        R_T_mask = (offs[:, None] < D) & (offs[None, :] < D)
        R_T_block = tl.load(R_T_ptr + R_T_offs, mask=R_T_mask, other=0.0).to(
            tl.float32
        )

        out_2d = tl.dot(rotated_2d, R_T_block)  # [1, BLOCK_D]
        out = tl.reshape(out_2d, [BLOCK_D])

        tl.store(
            output_ptr + row_idx * stride_output_row + offs,
            out.to(tl.float16),
            mask=mask,
        )

    @triton.jit
    def _turboquant_decode_gather_kernel(
        # Inputs
        codes_ptr,
        scale_ptr,
        outlier_ptr,
        outlier_mask_ptr,
        R_T_ptr,
        gather_idx_ptr,
        # QJL inputs (ignored if has_qjl is False)
        qjl_bits_ptr,
        qjl_norm_ptr,
        # Output
        output_ptr,
        # Strides
        stride_codes_row,
        stride_outlier_row,
        stride_RT_row,
        stride_output_row,
        stride_qjl_bits_row,
        # Params
        D: tl.constexpr,
        n_normal: tl.constexpr,
        n_outlier: tl.constexpr,
        N_LEVELS: tl.constexpr,
        has_qjl: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Gather source rows by indices + fused decode in one kernel."""
        row_idx = tl.program_id(0)
        src_row = tl.load(gather_idx_ptr + row_idx).to(tl.int32)

        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        scale = tl.load(scale_ptr + src_row).to(tl.float32)
        outlier_mask = tl.load(outlier_mask_ptr + offs, mask=mask, other=0).to(tl.int1)
        normal_mask = ~outlier_mask & mask

        normal_int = tl.where(normal_mask, 1, 0)
        normal_cumsum = tl.cumsum(normal_int, axis=0) - 1
        outlier_int = tl.where(outlier_mask & mask, 1, 0)
        outlier_cumsum = tl.cumsum(outlier_int, axis=0) - 1

        raw_codes = tl.load(
            codes_ptr + src_row * stride_codes_row + normal_cumsum,
            mask=normal_mask,
            other=0,
        ).to(tl.float32)
        dequant_normal = (raw_codes / (N_LEVELS - 1) * 2.0 - 1.0) * scale

        if has_qjl:
            sign_raw = tl.load(
                qjl_bits_ptr + src_row * stride_qjl_bits_row + normal_cumsum,
                mask=normal_mask,
                other=0,
            ).to(tl.float32)
            signs = sign_raw * 2.0 - 1.0
            res_norm = tl.load(qjl_norm_ptr + src_row).to(tl.float32)
            correction = signs * (res_norm * 1.2533141 / n_normal)
            dequant_normal = dequant_normal + correction

        outlier_vals = tl.load(
            outlier_ptr + src_row * stride_outlier_row + outlier_cumsum,
            mask=outlier_mask & mask,
            other=0.0,
        ).to(tl.float32)

        rotated = tl.zeros([BLOCK_D], dtype=tl.float32)
        rotated = tl.where(normal_mask, dequant_normal, rotated)
        rotated = tl.where(outlier_mask & mask, outlier_vals, rotated)

        rotated_2d = tl.reshape(rotated, [1, BLOCK_D])
        R_T_offs = offs[:, None] * stride_RT_row + offs[None, :]
        R_T_mask = (offs[:, None] < D) & (offs[None, :] < D)
        R_T_block = tl.load(R_T_ptr + R_T_offs, mask=R_T_mask, other=0.0).to(
            tl.float32
        )
        out_2d = tl.dot(rotated_2d, R_T_block)
        out = tl.reshape(out_2d, [BLOCK_D])
        tl.store(
            output_ptr + row_idx * stride_output_row + offs,
            out.to(tl.float16),
            mask=mask,
        )

    @triton.jit
    def _turboquant_decode_gather_packed4_kernel(
        # Inputs
        packed_codes_ptr,
        scale_ptr,
        outlier_ptr,
        outlier_mask_ptr,
        R_T_ptr,
        gather_idx_ptr,
        # Output
        output_ptr,
        # Strides
        stride_codes_row,
        stride_outlier_row,
        stride_RT_row,
        stride_output_row,
        # Params
        D: tl.constexpr,
        n_normal: tl.constexpr,
        n_outlier: tl.constexpr,
        N_LEVELS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused gather+decode for 4-bit packed codes (2 values/byte)."""
        row_idx = tl.program_id(0)
        src_row = tl.load(gather_idx_ptr + row_idx).to(tl.int32)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        scale = tl.load(scale_ptr + src_row).to(tl.float32)
        outlier_mask = tl.load(outlier_mask_ptr + offs, mask=mask, other=0).to(tl.int1)
        normal_mask = ~outlier_mask & mask

        normal_int = tl.where(normal_mask, 1, 0)
        normal_cumsum = tl.cumsum(normal_int, axis=0) - 1
        outlier_int = tl.where(outlier_mask & mask, 1, 0)
        outlier_cumsum = tl.cumsum(outlier_int, axis=0) - 1

        packed_idx = normal_cumsum // 2
        packed_byte = tl.load(
            packed_codes_ptr + src_row * stride_codes_row + packed_idx,
            mask=normal_mask,
            other=0,
        ).to(tl.uint8)

        is_high = (normal_cumsum % 2) == 1
        low_nibble = packed_byte & 0x0F
        high_nibble = (packed_byte >> 4) & 0x0F
        codes_u8 = tl.where(is_high, high_nibble, low_nibble).to(tl.float32)
        dequant_normal = (codes_u8 / (N_LEVELS - 1) * 2.0 - 1.0) * scale

        outlier_vals = tl.load(
            outlier_ptr + src_row * stride_outlier_row + outlier_cumsum,
            mask=outlier_mask & mask,
            other=0.0,
        ).to(tl.float32)

        rotated = tl.zeros([BLOCK_D], dtype=tl.float32)
        rotated = tl.where(normal_mask, dequant_normal, rotated)
        rotated = tl.where(outlier_mask & mask, outlier_vals, rotated)

        rotated_2d = tl.reshape(rotated, [1, BLOCK_D])
        R_T_offs = offs[:, None] * stride_RT_row + offs[None, :]
        R_T_mask = (offs[:, None] < D) & (offs[None, :] < D)
        R_T_block = tl.load(R_T_ptr + R_T_offs, mask=R_T_mask, other=0.0).to(
            tl.float32
        )
        out_2d = tl.dot(rotated_2d, R_T_block)
        out = tl.reshape(out_2d, [BLOCK_D])
        tl.store(
            output_ptr + row_idx * stride_output_row + offs,
            out.to(tl.float16),
            mask=mask,
        )


def turboquant_encode_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    R: torch.Tensor,
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    outlier_mask: torch.Tensor,
    bits: int = 4,
    use_qjl: bool = True,
) -> dict:
    """Encode KV tensors using Triton kernels. Falls back to PyTorch if Triton unavailable."""
    if not HAS_TRITON:
        from sglang.srt.layers.quantization.turboquant import turboquant_encode_v2

        return turboquant_encode_v2(
            k, v, R, codebook_k, codebook_v, outlier_mask, bits=bits, use_qjl=use_qjl
        )

    batch_shape = k.shape[:-1]
    D = k.shape[-1]
    n_outlier = outlier_mask.sum().item()
    n_normal = D - n_outlier
    n_levels = 2**bits

    result: dict = {}
    result["outlier_mask"] = outlier_mask
    result["codebook_k"] = codebook_k
    result["codebook_v"] = codebook_v

    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    for name, x, codebook in [("k", k, codebook_k), ("v", v, codebook_v)]:
        x_flat = x.reshape(-1, D).float().contiguous()
        n_rows = x_flat.shape[0]

        codes = torch.empty(n_rows, n_normal, dtype=torch.uint8, device=x.device)
        scale = torch.empty(n_rows, dtype=torch.float16, device=x.device)
        outliers = torch.empty(n_rows, n_outlier, dtype=torch.float16, device=x.device)

        R_cont = R.float().contiguous()
        om_cont = outlier_mask.to(x.device).contiguous()

        grid = (n_rows,)
        _turboquant_encode_kernel[grid](
            x_flat,
            R_cont,
            om_cont,
            codes,
            scale,
            outliers,
            # Strides
            x_flat.stride(0),
            R_cont.stride(0),
            codes.stride(0),
            outliers.stride(0),
            # Params
            D=D,
            n_normal=n_normal,
            n_outlier=n_outlier,
            N_LEVELS=n_levels,
            BLOCK_D=BLOCK_D,
        )

        result[f"{name}_codes"] = codes.reshape(*batch_shape, n_normal)
        result[f"{name}_scale"] = scale.reshape(*batch_shape, 1)
        result[f"{name}_outliers"] = outliers.reshape(*batch_shape, n_outlier)

    # QJL residual correction (still in PyTorch for encode — sign bits require
    # a random projection that doesn't benefit from Triton fusion)
    if use_qjl:
        from sglang.srt.layers.quantization.turboquant import qjl_encode_residual

        normal_mask = ~outlier_mask
        for name, x, codebook in [("k", k, codebook_k), ("v", v, codebook_v)]:
            x_rot = (x.float() @ R.float()).to(x.dtype)
            x_normal = x_rot[..., normal_mask]
            # Dequantize using uniform inverse (matches kernel quantization)
            recon_normal = (
                result[f"{name}_codes"].float() / (n_levels - 1) * 2.0 - 1.0
            ) * result[f"{name}_scale"].float()
            residual = x_normal.float() - recon_normal
            qjl_bits, qjl_norm = qjl_encode_residual(residual)
            result[f"{name}_qjl_bits"] = qjl_bits
            result[f"{name}_qjl_norm"] = qjl_norm

    return result


def turboquant_decode_triton(
    encoded: dict,
    R_T: torch.Tensor,
    use_qjl: bool = True,
) -> tuple:
    """Decode TurboQuant-encoded KV tensors using Triton. Falls back to PyTorch."""
    if not HAS_TRITON:
        from sglang.srt.layers.quantization.turboquant import turboquant_decode_v2

        return turboquant_decode_v2(encoded, R_T, use_qjl=use_qjl)

    outlier_mask = encoded["outlier_mask"]
    D = outlier_mask.shape[0]
    n_outlier = outlier_mask.sum().item()
    n_normal = D - n_outlier

    # Infer N_LEVELS from codebook size
    n_levels = encoded["codebook_k"].shape[0]

    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    R_T_cont = R_T.float().contiguous()
    om_cont = outlier_mask.to(R_T.device).contiguous()

    results = []
    for name in ["k", "v"]:
        codes = encoded[f"{name}_codes"]
        batch_shape = codes.shape[:-1]
        scale = encoded[f"{name}_scale"]
        outliers = encoded[f"{name}_outliers"]

        codes_flat = codes.reshape(-1, n_normal).contiguous()
        n_rows = codes_flat.shape[0]
        scale_flat = scale.reshape(-1).contiguous()
        outliers_flat = outliers.reshape(-1, n_outlier).contiguous()

        output = torch.empty(n_rows, D, dtype=torch.float16, device=codes.device)

        has_qjl = use_qjl and f"{name}_qjl_bits" in encoded

        # QJL tensors (provide dummy pointers if not used)
        if has_qjl:
            qjl_bits = encoded[f"{name}_qjl_bits"].reshape(-1, n_normal).contiguous()
            qjl_norm = encoded[f"{name}_qjl_norm"].reshape(-1).contiguous()
            stride_qjl_bits_row = qjl_bits.stride(0)
        else:
            qjl_bits = codes_flat  # dummy, won't be read
            qjl_norm = scale_flat  # dummy, won't be read
            stride_qjl_bits_row = 0

        grid = (n_rows,)
        _turboquant_decode_kernel[grid](
            codes_flat,
            scale_flat,
            outliers_flat,
            om_cont,
            R_T_cont,
            qjl_bits,
            qjl_norm,
            output,
            # Strides
            codes_flat.stride(0),
            outliers_flat.stride(0),
            R_T_cont.stride(0),
            output.stride(0),
            stride_qjl_bits_row,
            # Params
            D=D,
            n_normal=n_normal,
            n_outlier=n_outlier,
            N_LEVELS=n_levels,
            has_qjl=has_qjl,
            BLOCK_D=BLOCK_D,
        )

        results.append(output.reshape(*batch_shape, D))

    return results[0], results[1]


def turboquant_decode_selected_triton(
    codes_all: torch.Tensor,
    scale_all: torch.Tensor,
    outliers_all: torch.Tensor,
    outlier_mask: torch.Tensor,
    R_T: torch.Tensor,
    gather_indices: torch.Tensor,
    n_levels: int,
    use_qjl: bool = False,
    qjl_bits_all: Optional[torch.Tensor] = None,
    qjl_norm_all: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode selected rows with a fused gather+decode kernel."""
    if gather_indices.numel() == 0:
        return torch.empty((0, outlier_mask.shape[0]), dtype=torch.float16, device=codes_all.device)

    if not HAS_TRITON:
        indices = gather_indices.long()
        codes = codes_all[indices].float()
        scale = scale_all[indices].float()
        outliers = outliers_all[indices].float()
        normal_mask = ~outlier_mask
        n_normal = int(normal_mask.sum().item())
        recon = (codes / (n_levels - 1) * 2.0 - 1.0) * scale
        if use_qjl and qjl_bits_all is not None and qjl_norm_all is not None:
            signs = qjl_bits_all[indices].float() * 2.0 - 1.0
            recon = recon + signs * (qjl_norm_all[indices].float() * 1.2533141 / n_normal)
        d = outlier_mask.shape[0]
        out = torch.zeros((indices.numel(), d), dtype=torch.float32, device=codes_all.device)
        out[:, normal_mask] = recon
        out[:, outlier_mask] = outliers
        return (out @ R_T.float()).to(torch.float16)

    D = outlier_mask.shape[0]
    n_outlier = int(outlier_mask.sum().item())
    n_normal = D - n_outlier
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    codes_2d = codes_all.contiguous()
    scale_1d = scale_all.reshape(-1).contiguous()
    outliers_2d = outliers_all.contiguous()
    gather_idx = gather_indices.to(device=codes_all.device).contiguous()
    om = outlier_mask.to(codes_all.device).contiguous()
    R_T_cont = R_T.float().contiguous()
    output = torch.empty((gather_idx.numel(), D), dtype=torch.float16, device=codes_all.device)

    has_qjl = use_qjl and qjl_bits_all is not None and qjl_norm_all is not None
    if has_qjl:
        qjl_bits = qjl_bits_all.contiguous()
        qjl_norm = qjl_norm_all.reshape(-1).contiguous()
        stride_qjl_bits_row = qjl_bits.stride(0)
    else:
        qjl_bits = codes_2d
        qjl_norm = scale_1d
        stride_qjl_bits_row = 0

    _turboquant_decode_gather_kernel[(gather_idx.numel(),)](
        codes_2d,
        scale_1d,
        outliers_2d,
        om,
        R_T_cont,
        gather_idx,
        qjl_bits,
        qjl_norm,
        output,
        codes_2d.stride(0),
        outliers_2d.stride(0),
        R_T_cont.stride(0),
        output.stride(0),
        stride_qjl_bits_row,
        D=D,
        n_normal=n_normal,
        n_outlier=n_outlier,
        N_LEVELS=n_levels,
        has_qjl=has_qjl,
        BLOCK_D=BLOCK_D,
    )
    return output


def turboquant_decode_selected_packed4_triton(
    packed_codes_all: torch.Tensor,
    scale_all: torch.Tensor,
    outliers_all: torch.Tensor,
    outlier_mask: torch.Tensor,
    R_T: torch.Tensor,
    gather_indices: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """Decode selected rows for 4-bit packed codes with fused gather kernel."""
    if gather_indices.numel() == 0:
        return torch.empty((0, outlier_mask.shape[0]), dtype=torch.float16, device=packed_codes_all.device)

    d = outlier_mask.shape[0]
    n_outlier = int(outlier_mask.sum().item())
    n_normal = d - n_outlier
    if n_levels > 16:
        raise ValueError(f"packed4 decode only supports n_levels <= 16, got {n_levels}")

    if not HAS_TRITON:
        indices = gather_indices.long()
        packed = packed_codes_all[indices]
        out = torch.empty((*packed.shape[:-1], n_normal), dtype=torch.uint8, device=packed.device)
        out[..., 0::2] = packed & 0x0F
        out[..., 1::2] = (packed >> 4) & 0x0F
        codes = out[..., :n_normal].float()
        scale = scale_all[indices].float()
        recon = (codes / (n_levels - 1) * 2.0 - 1.0) * scale
        outliers = outliers_all[indices].float()
        full = torch.zeros((indices.numel(), d), dtype=torch.float32, device=packed.device)
        normal_mask = ~outlier_mask
        full[:, normal_mask] = recon
        full[:, outlier_mask] = outliers
        return (full @ R_T.float()).to(torch.float16)

    BLOCK_D = 1
    while BLOCK_D < d:
        BLOCK_D *= 2

    packed_codes = packed_codes_all.contiguous()
    scale_1d = scale_all.reshape(-1).contiguous()
    outliers_2d = outliers_all.contiguous()
    gather_idx = gather_indices.to(device=packed_codes_all.device).contiguous()
    om = outlier_mask.to(packed_codes_all.device).contiguous()
    R_T_cont = R_T.float().contiguous()
    output = torch.empty((gather_idx.numel(), d), dtype=torch.float16, device=packed_codes_all.device)

    _turboquant_decode_gather_packed4_kernel[(gather_idx.numel(),)](
        packed_codes,
        scale_1d,
        outliers_2d,
        om,
        R_T_cont,
        gather_idx,
        output,
        packed_codes.stride(0),
        outliers_2d.stride(0),
        R_T_cont.stride(0),
        output.stride(0),
        D=d,
        n_normal=n_normal,
        n_outlier=n_outlier,
        N_LEVELS=n_levels,
        BLOCK_D=BLOCK_D,
    )
    return output
