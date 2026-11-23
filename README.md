# pairwise-bone-palette-fence

## 领域

游戏、图形与多媒体

## 主题

骨骼调色板纹理

## 说明

分块上传、帧栅栏与降级权重重算。

初始状态下 `graphics/bone_palette_fence.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
