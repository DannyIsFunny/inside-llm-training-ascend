# 用 TileLang 复现 DeepSeekV4 算子

这个目录用 TileLang 在 Ascend 上实现 DeepSeekV4 中一些新算法相关的算子，记录实现、调试和验证过程。

我之前用 Ascend C 写过算子。这次尝试 TileLang，主要是想跟进 DeepSeek 与 Ascend 生态正在推进的算子编程方式，也借实际复现熟悉这套工具。

参考：[TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)、[DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels)。
