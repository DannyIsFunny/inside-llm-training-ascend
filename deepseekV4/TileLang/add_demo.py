# Source: https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/TileLang-Ascend%20Programming%20Guide.md
# Section 2.2: TileLang 算子 kernel 示例 (elementwise add)

import tilelang
import tilelang.language as T
from tilelang import jit

import torch

# Comment From DannyIsFunny: 
#  1. 这两个config 是 TileLang 的核心优化， 分别表示自动插入同步断点、自动编排内存排布。 
#     通过内部的Graph Pass 自动做这些静态优化，提升 TileLang 编译出的算子的性能
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Comment From DannyIsFunny:
# 2. 这两处是 NPU/CPU 算子的通用输入信息。 
#     M、N 表示实际的输入大小。 
#     block_M、block_N 表示 计算时的切片分块大小，因为实际输入一般比 AICORE 的最大输入要大。
M = 1024
N = 1024
block_M = 128
block_N = 128

VEC_NUM = 2

# Comment From DannyIsFunny: 
#  3. 第一层算子注册，编译注册。(jit) 
#     out_idx=[-1] ：表示 Kernel 的最后一个输入是算子的输出
#     @jit 装饰的函数： 输入一般是算子的 输入Shape、精度，计算时的分块大小 
#                      (这些输入信息如果发生变化可能触发算子重编译)
@jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M: int, N: int, block_M: int, block_N: int, dtype: str = 'float'):
    m_num = M // block_M
    n_num = N // block_N

    # Comment From DannyIsFunny
    # 4. 第二层算子注册，计算逻辑注册. (prim_func)
    #    A、B、C： 分别是加法的输入和输出
    #    cid 表示分块 的ID，(也可以理解为 cube core id)
    #              eg.如果 [M,N]=[100,100],[block_M,block_N]=[50,50]。 就可以分成4块，cid可取0-3
    #    vid 只能取 0 或 1 。（可以理解为 vector core id）。
    #              vid 是对Ascend 芯片的适配，达芬奇架构 内部的 CubeCore num: VectorCore = 1:2 
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            # 计算步骤1：在 UB 上申请内存
            #        Comment From DannyIsFunny: 
            #             这里是 TileLang 的一个易用点，调用 allocateShard 会根据上下文自动选择创建在 UB 还是 L1
            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            # 计算步骤2：讲输入 按块拷贝 待到 UB
            #        Comment From DannyIsFunny: 
            #            T.copy 需要指定源和目标，但不需要手动指定使用的具体 MTE
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            # 计算步骤3. 并行调用VECTOR执行加法计算
            #        Comment From DannyIsFunny: T.Parallel、TileLang 编译器功能自动将 + 映射到 Vector 上的 Add 算子
            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
            # 计算步骤4. 将输出拷贝到 输出内存 上
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return add_kernel

######################################################################################################################
# DEMO ： 创建并使用这个算子的例子 
######################################################################################################################
#  1. 创建算子
func = tile_add(M, N, block_M, block_N)
#  2. 构造输入
torch.manual_seed(0)
a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()
torch.npu.synchronize()
print("init successful!")
#  3. 调用算子做加法
c = func(a, b)
ref_c = a + b
# 4. 验证精度：比较 TileLang 和 PyTorch 在 NPU 上计算的加法结果
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")

