"""骨骼调色板纹理：分块上传 / 帧栅栏 / 代际延迟释放 / 跨块对齐 / 降级权重重算 / 确定性蒙皮。

六个缺陷与本模块入口的对应关系
--------------------------------
1. 调色板分块与容量查询 ......... ``capacity`` / ``chunk_split``
2. 上传与绘制的帧栅栏 ........... ``upload_fence`` / ``FenceTracker``
3. 纹理代际与延迟释放 ........... ``generation`` / ``reload`` / ``delayed_release``
4. 跨块 32 位索引对齐 ........... ``index_align`` / ``validation`` / ``audit``
5. 影响数降级时的权重重算 ....... ``weight_renorm`` / ``fallback``
6. 蒙皮矩阵的确定性归约 ......... ``matrix_reduce`` / ``determinism``

辅助入口：``metrics`` / ``profile``。

模块只依赖标准库，用 CPU 侧数据结构精确模拟 GPU 侧的时序与内存约束，
所有数值归约均采用固定顺序 + ``math.fsum``，保证多次运行逐位一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "capacity",
    "chunk_split",
    "upload_fence",
    "generation",
    "delayed_release",
    "index_align",
    "weight_renorm",
    "matrix_reduce",
    "validation",
    "audit",
    "fallback",
    "metrics",
    "determinism",
    "reload",
    "profile",
    # 状态对象（供各函数式入口串联使用）
    "Chunk",
    "FenceTracker",
    "FenceError",
    "TexturePool",
    "TextureGeneration",
]

# ---------------------------------------------------------------------------
# 常量：模拟 RGBA32F 调色板纹理的内存布局
# ---------------------------------------------------------------------------
FLOAT32_BYTES = 4                      # 32 位
MATRIX_FLOATS = 16                     # 4x4 行主序
MATRIX_BYTES = MATRIX_FLOATS * FLOAT32_BYTES   # 单个骨骼矩阵 64B
ALIGNMENT_BYTES = FLOAT32_BYTES                 # 采样要求的 32 位对齐
TEXELS_PER_MATRIX = 4                  # 每个矩阵占纹理一行（4 个 RGBA texel）
TEXTURE_WIDTH_TEXELS = TEXELS_PER_MATRIX

DEFAULT_UNIFORM_CAPACITY = 64          # uniform 数组能容纳的骨骼上限
DEFAULT_CHUNK_ROWS = 32                # 每个分块 32 行矩阵
GPU_FRAME_LATENCY = 2                  # GPU 滞后 CPU 的帧数
HIGH_END_INFLUENCES = 8                # 高端机每顶点最大骨骼影响数
LOW_END_INFLUENCES = 4                 # 低端机降级后的最大影响数

# profile 的确定性成本模型（不读墙钟，纯算术，保证可复现）
UPLOAD_COST_NS_PER_BYTE = 0.1
DRAW_COST_NS_PER_CHUNK = 5000.0
FENCE_WAIT_NS = 1000.0
RELEASE_COST_NS_PER_GENERATION = 2000.0

_IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FenceError(RuntimeError):
    """绘制命令引用了尚未被帧栅栏 signal 的分块。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _as_int(value: Any, name: str = "value") -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} 不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} 必须是整数，收到 {type(value).__name__}")


def _normalize_influences(
    influences: Any,
) -> list[tuple[int, float]]:
    """把多种形态的顶点影响归一为 ``[(bone_index, weight), ...]``。

    接受：``[(bone, w), ...]``、``{bone: w}``、``(indices, weights)``、``None``。
    """
    if influences is None:
        return []
    if isinstance(influences, Mapping):
        pairs = list(influences.items())
    elif (
        isinstance(influences, tuple)
        and len(influences) == 2
        and not isinstance(influences[0], (int, float))
        and isinstance(influences[0], Sequence)
    ):
        indices, weights = influences
        pairs = list(zip(indices, weights))
    elif isinstance(influences, Sequence):
        pairs = list(influences)
    else:
        raise TypeError("无法识别的影响数形态")
    return [(_as_int(b, "bone_index"), float(w)) for b, w in pairs]


# ---------------------------------------------------------------------------
# 修复 1：调色板分块与容量查询
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """调色板纹理的一个分块（纹理数组的一层）。"""

    chunk_id: int
    start_bone: int
    end_bone: int            # 不含
    rows: int                # 实际矩阵行数
    padding_rows: int        # 末尾补齐到 chunk_rows 的空行
    stride_rows: int         # rows + padding_rows
    byte_offset: int         # 在整纹理中的字节偏移
    byte_length: int         # 实际数据字节数
    stride_bytes: int        # 含 padding 的层跨度
    aligned: bool            # 偏移与跨度是否满足 32 位对齐


def chunk_split(
    bone_count: Any,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> list[Chunk]:
    """把 ``bone_count`` 个骨骼矩阵切成若干分块。

    关键不变量：分块永远按 *整矩阵* 切（每块行数是矩阵整数倍），
    矩阵占 64B、天然被 4B 整除，因此任何分块的偏移与跨度都 32 位对齐，
    单个矩阵不可能横跨两个分块——这是修复跨块采样问题的前提。
    """
    bone_count = _as_int(bone_count, "bone_count")
    chunk_rows = _as_int(chunk_rows, "chunk_rows")
    if bone_count < 0:
        raise ValueError("bone_count 不能为负")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows 必须为正")

    chunks: list[Chunk] = []
    for start in range(0, bone_count, chunk_rows):
        end = min(start + chunk_rows, bone_count)
        rows = end - start
        chunk_id = start // chunk_rows
        byte_offset = chunk_id * chunk_rows * MATRIX_BYTES
        byte_length = rows * MATRIX_BYTES
        stride_bytes = chunk_rows * MATRIX_BYTES
        aligned = (
            byte_offset % ALIGNMENT_BYTES == 0
            and byte_length % ALIGNMENT_BYTES == 0
            and stride_bytes % ALIGNMENT_BYTES == 0
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                start_bone=start,
                end_bone=end,
                rows=rows,
                padding_rows=chunk_rows - rows,
                stride_rows=chunk_rows,
                byte_offset=byte_offset,
                byte_length=byte_length,
                stride_bytes=stride_bytes,
                aligned=aligned,
            )
        )
    return chunks


def capacity(
    bone_count: Any,
    uniform_limit: int = DEFAULT_UNIFORM_CAPACITY,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """容量查询：骨骼数超 uniform 上限后改走调色板纹理，并给出分块布局。"""
    bone_count = _as_int(bone_count, "bone_count")
    uniform_limit = _as_int(uniform_limit, "uniform_limit")
    if bone_count < 0:
        raise ValueError("bone_count 不能为负")

    use_texture = bone_count > uniform_limit
    chunks = chunk_split(bone_count, chunk_rows) if use_texture else []
    return {
        "bone_count": bone_count,
        "uniform_limit": uniform_limit,
        "fits_in_uniform": not use_texture,
        "storage": "uniform" if not use_texture else "texture",
        "matrix_bytes": MATRIX_BYTES,
        "matrix_float_count": MATRIX_FLOATS,
        "texture_width_texels": TEXTURE_WIDTH_TEXELS,
        "texture_height_rows": bone_count,
        "chunk_rows": chunk_rows,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "data_bytes": bone_count * MATRIX_BYTES,
        "total_stride_bytes": sum(c.stride_bytes for c in chunks),
        "alignment_bytes": ALIGNMENT_BYTES,
        "aligned": all(c.aligned for c in chunks),
    }


# ---------------------------------------------------------------------------
# 修复 2：上传与绘制之间的帧栅栏
# ---------------------------------------------------------------------------
@dataclass
class FenceTracker:
    """跟踪每个分块最近一次上传所在帧，绘制前必须等到对应 fence signal。"""

    signaled_frames: set[int] = field(default_factory=set)
    signaled_generations: set[int] = field(default_factory=set)
    chunk_frame: dict[int, int] = field(default_factory=dict)
    pending: dict[int, list[int]] = field(default_factory=dict)
    draw_calls: list[dict[str, Any]] = field(default_factory=list)
    wait_events: list[dict[str, Any]] = field(default_factory=list)

    def submit_upload(self, chunk_ids: Iterable[int], frame: int) -> dict[str, Any]:
        """分块上传入队：记录该帧待 signal 的 fence，此时绘制还不允许引用。"""
        frame = _as_int(frame, "frame")
        ids = tuple(_as_int(c, "chunk_id") for c in chunk_ids)
        self.pending.setdefault(frame, []).extend(ids)
        for chunk_id in ids:
            self.chunk_frame[chunk_id] = frame
        return {"frame": frame, "chunks": ids, "signaled": False}

    def signal(
        self,
        frame: int,
        generation: int | None = None,
    ) -> bool:
        """GPU 完成该帧上传，fence 进入 signaled 状态。"""
        frame = _as_int(frame, "frame")
        if frame in self.pending:
            self.signaled_frames.add(frame)
        if generation is not None:
            self.signaled_generations.add(_as_int(generation, "generation"))
        return frame in self.signaled_frames

    def is_ready(self, chunk_ids: Iterable[int]) -> bool:
        return all(
            self.chunk_frame.get(c) in self.signaled_frames for c in chunk_ids
        )

    def wait_and_draw(self, chunk_ids: Iterable[int], frame: int) -> bool:
        """绘制入口：fence 未 signal 直接抛错，杜绝引用半成品纹理。"""
        frame = _as_int(frame, "frame")
        ids = tuple(_as_int(c, "chunk_id") for c in chunk_ids)
        missing = [
            c for c in ids if self.chunk_frame.get(c) not in self.signaled_frames
        ]
        if missing:
            raise FenceError(
                f"帧 {frame} 绘制引用了未 signal 的分块 {missing}："
                "上传与绘制之间缺少帧栅栏"
            )
        self.wait_events.append({"frame": frame, "chunks": ids})
        self.draw_calls.append({"frame": frame, "chunks": ids})
        return True

    @property
    def max_signaled_frame(self) -> int:
        return max(self.signaled_frames, default=-1)


def upload_fence(
    frame: Any,
    chunk_ids: Iterable[int] = (0,),
    tracker: FenceTracker | None = None,
) -> FenceTracker:
    """函数式入口：提交一帧分块上传并立即 signal，返回栅栏跟踪器。

    真实驱动中 submit 与 signal 分属 CPU/GPU 两侧；``TexturePool.reload``
    保留了两步分离的时序，本入口用于单步场景与测试。
    """
    tracker = tracker if tracker is not None else FenceTracker()
    record = tracker.submit_upload(chunk_ids, _as_int(frame, "frame"))
    tracker.signal(record["frame"])
    record["signaled"] = True
    return tracker


# ---------------------------------------------------------------------------
# 修复 3：纹理代际与延迟释放
# ---------------------------------------------------------------------------
@dataclass
class TextureGeneration:
    generation: int
    bone_count: int
    chunks: tuple[Chunk, ...]
    created_frame: int
    upload_frame: int
    storage: str


@dataclass
class TexturePool:
    """管理调色板纹理的多代际：旧纹理延迟到 GPU 不再引用后才释放。"""

    generation: int = 0
    textures: dict[int, TextureGeneration] = field(default_factory=dict)
    retire_queue: list[tuple[int, int]] = field(default_factory=list)  # (代际, 可释放帧)
    released: list[int] = field(default_factory=list)
    fence: FenceTracker = field(default_factory=FenceTracker)
    events: list[dict[str, Any]] = field(default_factory=list)

    def allocate(self, bone_count: int, frame: int = 0) -> TextureGeneration:
        cap = capacity(bone_count)
        gen = TextureGeneration(
            generation=self.generation,
            bone_count=bone_count,
            chunks=tuple(cap["chunks"]),
            created_frame=frame,
            upload_frame=frame,
            storage=cap["storage"],
        )
        self.textures[gen.generation] = gen
        self.events.append(
            {"type": "allocate", "generation": gen.generation, "frame": frame}
        )
        return gen

    def reload(
        self,
        bone_count: int,
        frame: int,
        latency: int = GPU_FRAME_LATENCY,
    ) -> TextureGeneration:
        """角色重载：新建代际 + 分块上传入队 + 旧代际排入延迟释放队列。"""
        old_generation = self.generation
        new_generation = old_generation + 1
        self.generation = new_generation

        cap = capacity(bone_count)
        gen = TextureGeneration(
            generation=new_generation,
            bone_count=bone_count,
            chunks=tuple(cap["chunks"]),
            created_frame=frame,
            upload_frame=frame,
            storage=cap["storage"],
        )
        self.textures[new_generation] = gen

        # 新代际分块上传：fence 尚未 signal，signal 前的绘制仍只能走旧代际
        chunk_ids = [c.chunk_id for c in gen.chunks] or [0]
        self.fence.submit_upload(chunk_ids, frame)

        # 旧代际不能立即销毁：上一帧的绘制命令可能仍在引用它，
        # 要等上传 fence signal 且再过 latency 帧（GPU 排空）后才可释放
        if old_generation in self.textures:
            self.retire_queue.append((old_generation, frame + latency))

        self.events.append(
            {
                "type": "reload",
                "old_generation": old_generation,
                "new_generation": new_generation,
                "frame": frame,
                "retire": [
                    (g, f) for g, f in self.retire_queue if g == old_generation
                ],
            }
        )
        return gen

    def signal_upload(self, frame: int) -> bool:
        return self.fence.signal(frame, generation=self.generation)

    def draw(
        self,
        frame: int,
        chunk_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        gen = self.textures[self.generation]
        ids = list(chunk_ids) if chunk_ids is not None else [
            c.chunk_id for c in gen.chunks
        ] or [0]
        self.fence.wait_and_draw(ids, frame)
        return {
            "frame": frame,
            "generation": self.generation,
            "chunks": tuple(ids),
        }

    def reap(self, frame: int) -> dict[str, Any]:
        return _reap_pool(self, frame)


def generation(start_generation: Any = 0) -> TexturePool:
    """创建纹理池（代际管理器）。"""
    return TexturePool(generation=_as_int(start_generation, "start_generation"))


def reload(
    target: Any,
    frame: int = 1,
    bone_count: int | None = None,
    latency: int = GPU_FRAME_LATENCY,
) -> TexturePool:
    """函数式重载入口。

    传入 ``TexturePool`` 时在其上重载；传入整数时视为骨骼数，
    自动先建一个含旧代际的池，再执行一次重载。
    """
    if isinstance(target, TexturePool):
        pool = target
        if bone_count is None:
            raise ValueError("在已有 TexturePool 上 reload 必须提供 bone_count")
    else:
        bone_count = _as_int(target, "bone_count")
        pool = generation(0)
        old = pool.allocate(bone_count, frame=0)
        # 旧代际的上传早已完成
        pool.fence.submit_upload(
            [c.chunk_id for c in old.chunks] or [0], frame=0
        )
        pool.fence.signal(0)
    pool.reload(bone_count, _as_int(frame, "frame"), latency)
    return pool


def _reap_pool(pool: TexturePool, frame: int) -> dict[str, Any]:
    ready: list[int] = []
    pending: list[tuple[int, int]] = []
    for gen_id, release_frame in pool.retire_queue:
        # 旧代际可释放的两个充要条件：
        # 1. 到达释放帧（GPU 帧延迟已排空）；
        # 2. 替换它的新代际的上传 fence 已 signal——新代际未 signal 前，
        #    在途绘制命令仍引用旧纹理，禁止释放。
        successor = gen_id + 1
        fence_ok = successor in pool.fence.signaled_generations
        if release_frame <= frame and fence_ok:
            pool.textures.pop(gen_id, None)
            pool.released.append(gen_id)
            ready.append(gen_id)
            pool.events.append(
                {"type": "release", "generation": gen_id, "frame": frame}
            )
        else:
            pending.append((gen_id, release_frame))
    pool.retire_queue = pending
    return {
        "frame": frame,
        "released": ready,
        "pending": pending,
        "live_generations": sorted(pool.textures),
    }


def delayed_release(target: Any, frame: Any = None) -> dict[str, Any]:
    """回收到达释放帧、且其上传 fence 已 signal 的旧代际。"""
    if isinstance(target, TexturePool):
        pool = target
        if frame is None:
            raise ValueError("延迟释放需要当前帧号")
        current_frame = _as_int(frame, "frame")
    else:
        # 单整数调用：空池上的回收演练，没有待释放代际
        pool = generation(0)
        current_frame = _as_int(target if frame is None else frame, "frame")
    return _reap_pool(pool, current_frame)


# ---------------------------------------------------------------------------
# 修复 4：跨块 32 位索引对齐
# ---------------------------------------------------------------------------
def index_align(
    bone_index: Any,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    bone_count: int | None = None,
) -> dict[str, Any]:
    """全局骨骼索引 -> (分块, 块内行, texel/字节偏移)，并校验对齐与跨块。

    因为分块只按整矩阵切且每矩阵 64B，块内偏移恒为 64B 的整数倍，
    自然满足 32 位对齐，矩阵也绝不会横跨两个分块。
    """
    idx = _as_int(bone_index, "bone_index")
    chunk_rows = _as_int(chunk_rows, "chunk_rows")
    if idx < 0:
        raise ValueError("bone_index 不能为负")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows 必须为正")
    if bone_count is not None and idx >= _as_int(bone_count, "bone_count"):
        raise IndexError(f"骨骼索引 {idx} 越界（共 {bone_count} 块骨骼）")

    chunk_id, local_row = divmod(idx, chunk_rows)
    byte_offset = idx * MATRIX_BYTES
    chunk_byte_offset = local_row * MATRIX_BYTES
    aligned = (
        byte_offset % ALIGNMENT_BYTES == 0
        and chunk_byte_offset % ALIGNMENT_BYTES == 0
    )
    # 整矩阵切分下 local_row 恒小于 chunk_rows，不可能跨块；保留显式守卫
    straddles_chunk = local_row + 1 > chunk_rows
    if straddles_chunk or not aligned:
        raise ValueError(
            f"骨骼 {idx} 的采样跨块或未按 32 位对齐："
            f"chunk={chunk_id} row={local_row} offset={byte_offset}"
        )

    return {
        "bone_index": idx,
        "chunk_id": chunk_id,
        "local_row": local_row,
        "texel_coord": (0, local_row),
        "texel_span": TEXELS_PER_MATRIX,
        "byte_offset": byte_offset,
        "chunk_byte_offset": chunk_byte_offset,
        "alignment_bytes": ALIGNMENT_BYTES,
        "aligned": True,
        "straddles_chunk": False,
    }


def resolve_indices(
    bone_indices: Iterable[int],
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    bone_count: int | None = None,
) -> list[dict[str, Any]]:
    """批量解析一个顶点引用的骨骼索引，任一跨块/错位即抛错。"""
    return [
        index_align(idx, chunk_rows, bone_count) for idx in bone_indices
    ]


# ---------------------------------------------------------------------------
# 修复 5：影响数降级时的权重重算
# ---------------------------------------------------------------------------
def weight_renorm(
    influences: Any,
    max_influences: int = LOW_END_INFLUENCES,
) -> dict[str, Any]:
    """截断到前 ``max_influences`` 个影响并重新归一化权重。

    确定性保证：先按权重降序、骨骼索引升序（平局兜底）排序，
    再用 ``math.fsum`` 求总和后逐项相除——与输入顺序无关、逐位可复现。
    低端机从 8 影响降到 4 影响时被丢弃的权重质量必须重新分配给保留项，
    否则剩余权重之和小于 1，蒙皮会出现收缩偏差。
    """
    if isinstance(influences, int):
        # 配置式调用：weight_renorm(4) 只表达降级档位
        max_influences = influences
        influences = None
    max_influences = _as_int(max_influences, "max_influences")
    if max_influences <= 0:
        raise ValueError("max_influences 必须为正")

    pairs = _normalize_influences(influences)
    ordered = sorted(pairs, key=lambda bw: (-bw[1], bw[0]))
    kept = ordered[:max_influences]
    dropped = ordered[max_influences:]

    total = math.fsum(w for _, w in kept)
    if total > 0.0:
        renorm = [(bone, w / total) for bone, w in kept]
        weight_sum = math.fsum(w for _, w in renorm)
    else:
        renorm = []
        weight_sum = 0.0

    return {
        "influences": renorm,
        "dropped": dropped,
        "max_influences": max_influences,
        "original_count": len(pairs),
        "kept_count": len(renorm),
        "weight_sum": weight_sum,
        "normalized": bool(renorm) and abs(weight_sum - 1.0) <= 1e-12,
    }


def fallback(influences: Any, target_influences: int = LOW_END_INFLUENCES) -> dict[str, Any]:
    """低端机降级入口：8 影响 -> 4（或 2）影响，权重同步重算。"""
    if isinstance(influences, int):
        target_influences = influences
        influences = None
    target_influences = _as_int(target_influences, "target_influences")
    result = weight_renorm(influences, target_influences)
    result["tier"] = "low" if target_influences <= LOW_END_INFLUENCES else "high"
    result["downgraded"] = result["original_count"] > target_influences
    return result


# ---------------------------------------------------------------------------
# 修复 6：蒙皮矩阵的确定性归约
# ---------------------------------------------------------------------------
def _lookup_matrix(matrices: Any, bone: int) -> Sequence[Sequence[float]]:
    if isinstance(matrices, Mapping):
        if bone not in matrices:
            raise KeyError(f"缺少骨骼 {bone} 的矩阵")
        matrix = matrices[bone]
    else:
        matrix = matrices[bone]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("蒙皮矩阵必须是 4x4")
    return matrix


def matrix_reduce(matrices: Any, influences: Any = None) -> list[list[float]]:
    """蒙皮矩阵归约：``M = Σ_k w_k · M_{i_k}``。

    确定性来源：
    1. 权重先经 ``weight_renorm`` 用 ``fsum`` 归一；
    2. 累加项按骨骼索引升序固定排列，不依赖输入/哈希顺序；
    3. 每个输出元素独立用 ``math.fsum``（正确舍入求和）归约。
    同一动画两次运行得到逐位一致的结果。
    """
    if isinstance(matrices, int):
        # 便捷形态：n 个单位矩阵等权归约
        n = _as_int(matrices, "matrix_count")
        if n < 0:
            raise ValueError("矩阵数不能为负")
        matrices = [_IDENTITY_4X4] * n
        influences = [(i, 1.0 / n) for i in range(n)] if n else []

    if influences is None:
        count = len(matrices)
        influences = [(i, 1.0 / count) for i in range(count)] if count else []

    renorm = weight_renorm(influences, max_influences=len(_normalize_influences(influences)) or 1)
    terms = sorted(renorm["influences"], key=lambda bw: bw[0])

    out: list[list[float]] = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            out[row][col] = math.fsum(
                weight * float(_lookup_matrix(matrices, bone)[row][col])
                for bone, weight in terms
            )
    return out


def determinism(matrices: Any, influences: Any = None, runs: int = 2) -> dict[str, Any]:
    """重复归约 ``runs`` 次并逐元素比对，验证蒙皮结果的确定性。"""
    runs = _as_int(runs, "runs")
    if runs < 2:
        raise ValueError("runs 至少为 2")
    first = matrix_reduce(matrices, influences)
    outputs = [first] + [
        matrix_reduce(matrices, influences) for _ in range(runs - 1)
    ]
    deviations = [
        (run, row, col)
        for run, candidate in enumerate(outputs[1:], start=2)
        for row in range(4)
        for col in range(4)
        if candidate[row][col] != first[row][col]
    ]
    return {
        "bit_identical": not deviations,
        "runs": runs,
        "deviations": deviations,
        "matrix": first,
    }


# ---------------------------------------------------------------------------
# 校验、审计、指标、画像
# ---------------------------------------------------------------------------
def _build_demo_pipeline(bone_count: int, frame: int = 0) -> TexturePool:
    """构造一条健康的默认管线（已上传、已 signal），供整数式调用使用。"""
    pool = generation(0)
    gen = pool.allocate(bone_count, frame=frame)
    ids = [c.chunk_id for c in gen.chunks] or [0]
    pool.fence.submit_upload(ids, frame)
    pool.fence.signal(frame)
    return pool


def validation(
    target: Any,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    influences: Any = None,
) -> dict[str, Any]:
    """静态校验：容量路径、分块对齐、索引跨块、权重归一化。"""
    if isinstance(target, TexturePool):
        bone_count = max(
            (g.bone_count for g in target.textures.values()), default=0
        )
    else:
        bone_count = _as_int(target, "bone_count")

    issues: list[str] = []
    cap = capacity(bone_count, chunk_rows=chunk_rows)
    for chunk in cap["chunks"]:
        if not chunk.aligned:
            issues.append(f"分块 {chunk.chunk_id} 未按 {ALIGNMENT_BYTES}B 对齐")
        if chunk.byte_offset % ALIGNMENT_BYTES != 0:
            issues.append(f"分块 {chunk.chunk_id} 字节偏移错位")

    for idx in range(bone_count):
        loc = index_align(idx, chunk_rows, bone_count)
        if loc["straddles_chunk"]:
            issues.append(f"骨骼索引 {idx} 横跨分块边界")

    if influences is not None:
        renorm = weight_renorm(
            influences,
            max_influences=len(_normalize_influences(influences)) or 1,
        )
        if renorm["influences"] and not renorm["normalized"]:
            issues.append(
                f"权重和为 {renorm['weight_sum']!r}，未归一化到 1.0"
            )
        for bone, weight in renorm["influences"]:
            if weight < 0.0:
                issues.append(f"骨骼 {bone} 权重为负：{weight!r}")

    return {"ok": not issues, "issues": issues, "capacity": cap}


def audit(target: Any, frame: int = 0) -> dict[str, Any]:
    """运行时审计：栅栏违规、代际泄漏、对齐与权重问题汇总。"""
    if isinstance(target, TexturePool):
        pool = target
    else:
        pool = _build_demo_pipeline(_as_int(target, "bone_count"))
    frame = _as_int(frame, "frame")
    issues: list[str] = []

    # 1. 任何已记录的绘制都必须引用 signaled 的分块
    for draw in pool.fence.draw_calls:
        for chunk_id in draw["chunks"]:
            if pool.fence.chunk_frame.get(chunk_id) not in pool.fence.signaled_frames:
                issues.append(
                    f"帧 {draw['frame']} 绘制引用了未 signal 的分块 {chunk_id}"
                )

    # 2. 到达释放帧却仍存活的旧代际 = 延迟释放缺失（显存泄漏）
    for gen_id, release_frame in pool.retire_queue:
        if release_frame <= frame and gen_id in pool.textures:
            issues.append(f"代际 {gen_id} 已到释放帧 {release_frame} 仍未释放")

    # 3. 当前代际的分块对齐
    for gen in pool.textures.values():
        for chunk in gen.chunks:
            if not chunk.aligned:
                issues.append(
                    f"代际 {gen.generation} 分块 {chunk.chunk_id} 对齐失败"
                )

    return {
        "ok": not issues,
        "issues": issues,
        "frame": frame,
        "live_generations": sorted(pool.textures),
        "released_generations": list(pool.released),
        "pending_retires": list(pool.retire_queue),
        "fence_signaled_frames": sorted(pool.fence.signaled_frames),
        "draw_calls": len(pool.fence.draw_calls),
    }


def metrics(target: Any, frame: int = 0) -> dict[str, Any]:
    """管线计数器：字节量、分块数、栅栏等待、代际存活/释放。"""
    if isinstance(target, TexturePool):
        pool = target
    else:
        pool = _build_demo_pipeline(_as_int(target, "bone_count"))

    live = sorted(pool.textures.values(), key=lambda g: g.generation)
    data_bytes = sum(g.bone_count * MATRIX_BYTES for g in live)
    stride_bytes = sum(
        chunk.stride_bytes for g in live for chunk in g.chunks
    )
    chunk_count = sum(len(g.chunks) for g in live)
    return {
        "frame": frame,
        "live_generations": len(live),
        "released_generations": len(pool.released),
        "chunk_count": chunk_count,
        "data_bytes": data_bytes,
        "stride_bytes": stride_bytes,
        "padding_bytes": stride_bytes - data_bytes,
        "upload_frames": len(pool.fence.pending),
        "signaled_frames": len(pool.fence.signaled_frames),
        "fence_waits": len(pool.fence.wait_events),
        "draw_calls": len(pool.fence.draw_calls),
        "pending_retires": len(pool.retire_queue),
    }


def profile(target: Any, frame: int = 0) -> dict[str, Any]:
    """确定性成本画像：基于字节量与分块数的固定成本模型，不读墙钟。"""
    if isinstance(target, TexturePool):
        pool = target
    else:
        pool = _build_demo_pipeline(_as_int(target, "bone_count"))

    m = metrics(pool, frame)
    upload_ns = m["stride_bytes"] * UPLOAD_COST_NS_PER_BYTE
    draw_ns = m["draw_calls"] * DRAW_COST_NS_PER_CHUNK
    fence_ns = m["fence_waits"] * FENCE_WAIT_NS
    release_ns = m["released_generations"] * RELEASE_COST_NS_PER_GENERATION
    total_ns = upload_ns + draw_ns + fence_ns + release_ns
    return {
        "frame": frame,
        "upload_ns": upload_ns,
        "draw_ns": draw_ns,
        "fence_ns": fence_ns,
        "release_ns": release_ns,
        "total_ns": total_ns,
        "metrics": m,
    }


# ---------------------------------------------------------------------------
# 自测演示：把六个修复点串成一条完整时序
# ---------------------------------------------------------------------------
def _demo() -> None:
    # 1. 容量查询与分块：200 块骨骼超出 64 的 uniform 上限 -> 纹理路径
    cap = capacity(200)
    assert cap["storage"] == "texture" and cap["chunk_count"] == 7
    assert cap["aligned"]

    # 2. 跨块索引对齐：最后一块骨骼仍落在唯一分块内且 32 位对齐
    loc = index_align(199, bone_count=200)
    assert loc["chunk_id"] == 6 and loc["aligned"] and not loc["straddles_chunk"]

    # 3. 帧栅栏：未 signal 时绘制必须被拒绝
    fence = FenceTracker()
    fence.submit_upload([0, 1], frame=10)
    try:
        fence.wait_and_draw([0], frame=10)
    except FenceError:
        pass
    else:
        raise AssertionError("未 signal 的分块不应允许绘制")
    fence.signal(10)
    assert fence.wait_and_draw([0, 1], frame=10)

    # 4. 代际重载与延迟释放
    pool = reload(200, frame=10)           # 代际 0 -> 1，旧代际排入延迟释放
    try:
        pool.draw(10)                      # 新代际 fence 未 signal
    except FenceError:
        pass
    else:
        raise AssertionError("重载后新代际未 signal 前不应可绘制")
    pool.signal_upload(10)
    assert pool.draw(10)["generation"] == 1
    assert delayed_release(pool, 11)["released"] == []      # GPU 尚未排空
    assert delayed_release(pool, 12)["released"] == [0]     # 到帧释放

    # 5. 降级权重重算：8 影响 -> 4 影响，权重和重新归一到 1
    vertices = [
        [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.05),
         (4, 0.03), (5, 0.01), (6, 0.005), (7, 0.005)]
    ]
    low = fallback(vertices[0], LOW_END_INFLUENCES)
    assert low["kept_count"] == 4 and abs(low["weight_sum"] - 1.0) < 1e-12

    # 6. 确定性归约：打乱输入顺序，结果逐位一致
    matrices = {
        b: tuple(
            tuple(float(b + r * 4 + c) for c in range(4)) for r in range(4)
        )
        for b in range(4)
    }
    order_a = [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]
    order_b = list(reversed(order_a))
    assert matrix_reduce(matrices, order_a) == matrix_reduce(matrices, order_b)
    assert determinism(matrices, order_a, runs=5)["bit_identical"]

    assert audit(pool, frame=12)["ok"]
    assert validation(200, influences=order_a)["ok"]
    print("all six fixes demoed OK")


if __name__ == "__main__":
    _demo()
