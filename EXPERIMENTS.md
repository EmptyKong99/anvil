# anvil 实验记录

技术性实验日志(跟代码放一起)。人看的周报在 `PhD/Project/KernelAgent/`。

---

## EXP-001 · PTX skill 注入消融(2026-06-17)

**问题:** 把我们从 v8 蒸馏的 PTX 配方注入 prompt,能不能让一个便宜模型(DeepSeek)
写出更好的 gemm_bf16_nt?(= 量化"脚手架价值")

**设置:** anvil Route B loop(模型一次性出 kernel,无工具),DeepSeek v4-pro,
每边 6 轮,gemm_bf16_nt / required_5,RTX 5090(baseline@GPU6 / skill@GPU7)。
**唯一变量** = `--inject-skill`(把 `prompts.PTX_GEMM_SKILL` 加进 system)。
skill 内容 = mma.sync.m16n8k16 + ldmatrix 配方 + 累加器布局 + `.trans` 坑(源自我们
5090 实测的 v7/v8)。

**结果:**

| | baseline(无 skill) | skill(注入) |
|---|---|---|
| 6 轮里对的 | 1(iter5) | 1(iter5) |
| **最佳 geomean** | **0.0305x** | **0.5770x** |
| 那个对的 kernel 写法 | 放弃张量核,纯 SIMT tiled | 裸 ldmatrix + mma.sync,遵循 recipe,no `.trans` |

baseline best note 原话:"改用纯 SIMT tiled kernel,**避开所有 wmma/ptx 复杂性**"。
skill best note 原话:"用裸 ldmatrix 和 mma.sync **遵循 expert recipe**,B 行存当列存 K×N(**no .trans**)"。

**结论:**
1. **强正信号(机制层面,非运气):** 同模型/同预算,注入 PTX 知识的那版**真的照配方
   写了张量核 kernel(0.577x,≈我们 v2-v3 水平),连 `.trans` 坑都避开了**;没知识的
   直接躺平写朴素 SIMT(0.03x)。注入的知识被实际使用 = 脚手架确实起作用。delta ≈ 19x。
2. **⚠️ 统计上不稳:** 每边只有 1 个对的(5/6 失败),单 best 对比方差大。**要坐实需每边
   重复 3–5 次看分布**,目前只能算"强 suggestive"。
3. **失败率高的原因:**(a)裸 PTX 张量核是模型最弱处;(b)Route B 一次性出码、不能自测;
   (c)**反馈被截断、且有些是 okbench 的 python traceback 而非 nvcc 报错** → 模型常看不到
   真错误、改不动。**修反馈截断两边都会涨。**

**下一步:** ① 每边重复 3–5 次,把 n=1 变成分布;② 修反馈(完整/head+tail 报错,且要给
nvcc 真报错而不是 okbench traceback);③ 之后再扩 skill 内容并重测。

**产物:** 服务器 `runs_ab/{base,skill}/gemm_bf16_nt_20260617_161943/`(kernel + results.jsonl + best.cu)。
