# anvil 实验记录

技术性实验日志(跟代码放一起)。人看的周报在 `PhD/Project/KernelAgent/`。
每个实验末尾附**逐 attempt 矩阵**(行=run,列=step,每行是一条 run 从左往右的轨迹):格子 = 加速比(geomean vs cuBLAS,正确) · `xC`=编译/校验失败 · `xW`=结果错 · `.`=未跑;行名 b=base(no-skill)/s=skill + rep 号。由 `collect_results.py` 读 results.jsonl 生成。

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

**逐 attempt 矩阵:**

| run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| b1 | xC | xW | xC | xC | 0.031 | xC |
| s1 | xC | xC | xW | xC | 0.577 | xC |

**产物:** `runs_exp001/{base,skill}/gemm_bf16_nt_20260617_161943/`(kernel + results.jsonl + best.cu;原名 `runs_ab`,已统一为 `runs_expNNN`)。

---

## EXP-002 · Agent(tool-loop) 上的 PTX skill 消融(2026-06-18)

**问题:** 把 anvil 从 Route B(一次性出码)换成 **Route-AVO-lite agent**(模型自带
`bench_kernel` 工具,自己编译→看真报错→改→迭代),再做同一个 skill 消融。EXP-001
的弱 claim("给没手、没知识的模型喂 PTX 配方")升级为强 claim:**给一个能自主迭代到
天花板的 agent 注入 skill,还能不能更高/更稳。**

**设置:** `anvil agent`(本会话新建,见 `agent.py`),DeepSeek v4-pro,`--max-attempts 12`,
gemm_bf16_nt / required_5,RTX 5090。**两臂各 n=3**:base=no-skill@GPU6,
skill=`--inject-skill`@GPU7,**唯一变量仍是 skill 注入**。skill 内容同 EXP-001
(mma.sync.m16n8k16 + ldmatrix 配方 + `.trans` 坑,源自我们 v7/v8 实测)。

**结果(每 run 的 best geomean vs cuBLAS):**

| | rep1 | rep2 | rep3 | 编译失败(/12) | 中位数 |
|---|---|---|---|---|---|
| base(no-skill) | 0.783 | 0.113 | 0.060 | 3 / 5 / 9 | 0.113 |
| **skill** | 0.746 | **0.883** | 0.060 | 0 / 2 / 6 | **0.746** |

**结论:**
1. **Route-AVO-lite 成立、且明显强于 Route B:** agent 真的"用手"——会读真报错、修
   正确性 bug(验证 run 里 attempt1→2 自己诊断出索引 double-count 并改对)、爬性能。
   EXP-001 的 Route B 同模型只到 ~0.06–0.58 且 5/6 失败;agent 两臂都有 run 爬到 ~0.78–0.88。
2. **skill 的两个正向作用(机制可见):**(a)**推上张量核**——skill rep2 自主爬到
   **0.883 ≈ forge 当年 wmma 天花板**,rep1 稳到 0.746 且 0 编译失败;(b)**减少
   thrashing**——编译失败 base 17/36 vs skill 8/36(≈腰斩),正是 skill 给对 mma/ldmatrix
   配方 + 避 `.trans` 的预期收益。中位数 base 0.113 vs skill 0.746。
3. **⚠️ 仍是强 suggestive,非定论:** n=3、方差大,**两臂各有一个 0.06 的 dud**
   (base rep3 / skill rep3 都在 wmma 编译错上栽了 6–9 次、没爬起来)。要坐实需加 rep。

**观察到的两个 agent 失败模式(下一步要治):**
- **过峰回退:** base rep1 在 a5 摸到 0.783 后自己越改越烂(0.618→0.108)——agent 从
  "上一版"接着改、不回退到 best。反馈已带 best-so-far,但需更强的"低于最好版就回退"约束。
- **wmma 编译 thrashing:** dud run 把预算烧在张量核 API 编译错上。skill 减轻但未根除。

**下一步:** ① 加 rep(n≥5)把 dud 的影响摊平、给出分布;② 治"过峰回退"(显式 revert-to-best);
③ held-out shape 查过拟合;④ 之后再上 Claude agent 臂。

**逐 attempt 矩阵:**

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | xC | xW | 0.204 | xC | 0.783 | 0.618 | 0.464 | 0.311 | 0.108 | 0.319 | xC | 0.568 |
| b2 | 0.113 | xC | xW | xC | xW | xW | xC | xC | 0.113 | xW | 0.060 | xC |
| b3 | xC | xC | 0.058 | xC | 0.060 | xC | xC | xC | 0.059 | xC | xC | xC |
| s1 | 0.327 | 0.563 | xW | 0.568 | 0.545 | 0.648 | 0.680 | 0.716 | xW | 0.573 | 0.486 | 0.746 |
| s2 | 0.343 | xC | 0.759 | 0.711 | 0.741 | 0.880 | 0.823 | xC | 0.881 | 0.883 | 0.825 | 0.169 |
| s3 | xC | xC | xC | 0.050 | xW | xC | xW | xW | xC | 0.060 | xC | xW |

**产物:** 服务器 `runs_exp002/{base,skill}/gemm_bf16_nt_*/`(每 attempt kernel + results.jsonl + best.cu + summary.json)。
