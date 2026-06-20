# anvil 实验记录

技术性实验日志(跟代码放一起)。人看的周报在 `PhD/Project/KernelAgent/`。
每个实验末尾附**逐 attempt 矩阵**(行=run,列=step,每行是一条 run 从左往右的轨迹):格子 = **对 cuBLAS 的百分比**(100=持平),**加粗=该 run 峰值** · `xC`=编译/校验失败 · `xW`=结果错 · `.`=未跑;行名 base-/skill- + rep 号。由 `collect_results.py` 读 results.jsonl 生成。

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
| base-1 | xC | xW | xC | xC | **3.1%** | xC |
| skill-1 | xC | xC | xW | xC | **57.7%** | xC |

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
| base-1 | xC | xW | 20.4% | xC | **78.3%** | 61.8% | 46.4% | 31.1% | 10.8% | 31.9% | xC | 56.8% |
| base-2 | **11.3%** | xC | xW | xC | xW | xW | xC | xC | 11.3% | xW | 6.0% | xC |
| base-3 | xC | xC | 5.8% | xC | **6.0%** | xC | xC | xC | 5.9% | xC | xC | xC |
| skill-1 | 32.7% | 56.3% | xW | 56.8% | 54.5% | 64.8% | 68.0% | 71.6% | xW | 57.3% | 48.6% | **74.6%** |
| skill-2 | 34.3% | xC | 75.9% | 71.1% | 74.1% | 88.0% | 82.3% | xC | 88.1% | **88.3%** | 82.5% | 16.9% |
| skill-3 | xC | xC | xC | 5.0% | xW | xC | xW | xW | xC | **6.0%** | xC | xW |

**产物:** 服务器 `runs_exp002/{base,skill}/gemm_bf16_nt_*/`(每 attempt kernel + results.jsonl + best.cu + summary.json)。

---

## EXP-003 · 修好的 agent(revert-to-best + stuck-handling)+ n=5(2026-06-20)

**问题:** P1 给 agent 加了两招治 EXP-002 的失败模式——**revert-to-best**(低于最好版就回到 best 只改一处,治"过峰回退")+ **stuck-handling**(连续 3 次失败就让它退守 SIMT 锁正确性,治"wmma thrashing")——并把 n 从 3 加到 5。验证:失败模式压住没、结论更稳没。

**设置:** 同 EXP-002(DeepSeek agent,max-attempts 12,base@GPU6 / skill@GPU7,`--inject-skill` 唯一变量);改动 = 修好的 agent + n=5。

**结果(best per rep,% of cuBLAS):**

| 臂 | rep1–5 | 中位 | 均值 | ≥70% 的 rep | 正确 attempts | 编译失败 |
|---|---|---|---|---|---|---|
| base | 36.8 / 12.9 / 6.0 / 11.8 / 5.9 | 11.8 | 14.7 | 0/5 | 23/60 | 20/60 |
| **skill** | **82.2 / 18.5 / 84.5 / 77.9 / 77.1** | **77.9** | **68.0** | **4/5** | 43/60 | 9/60 |

**结论:**
1. **结论更稳、更干净:** skill **4/5 reps 爬到 77–84%**(中位 77.9),base **全部 ≤37%(中位 11.8、0/5 过 70)**。EXP-002 里 base-1 那个 78.3% 的高分,n=5 看就是**运气离群**——no-skill agent 极少自主破张量核,skill 才是把它推上去的原因。
2. **两个失败模式都缓解(机制可见):**
   - **过峰回退↓**:多数 skill run **收在峰值附近**(skill-1 收 82.2=峰、skill-5 收 77.1=峰、skill-4 收 76.0≈峰),不再像 EXP-002 skill-2 那样 88→17 崩 → revert-to-best 起效。
   - **thrashing↓**:编译失败率 base 47%→33%、skill 22%→15%;skill 正确率 61%→72% → stuck-handling 起效。
3. **⚠️ 没根治:** skill 仍有 1 个 dud(skill-2,18.5%,wmma 编译反复栽)。要更稳得继续 P2(profiling-in-loop)/ edit-not-rewrite。

**产物:** `runs_exp003/{base,skill}/gemm_bf16_nt_2026062*/`。

**逐 attempt 矩阵:**

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-1 | 18.5% | xC | xW | xC | xC | 18.5% | 16.7% | 19.0% | 19.3% | xC | 24.2% | **36.8%** |
| base-2 | xC | 12.3% | xC | xC | xC | 9.4% | 12.4% | xC | **12.9%** | 9.3% | xC | xC |
| base-3 | 6.0% | xC | xC | 3.2% | 5.7% | xW | xW | **6.0%** | xW | xW | xW | 6.0% |
| base-4 | 1.1% | xC | xC | xC | xW | xC | **11.8%** | xW | xW | xW | 11.3% | xW |
| base-5 | xC | **5.9%** | xC | xW | 5.9% | xC | xW | 5.8% | xW | xW | xW | xW |
| skill-1 | xW | 43.6% | 76.8% | 55.8% | 76.3% | 47.3% | 60.6% | 2.5% | 78.7% | 79.6% | xC | **82.2%** |
| skill-2 | xW | xW | xC | 12.3% | xC | xW | 18.2% | xC | 18.2% | xW | **18.5%** | xC |
| skill-3 | 40.1% | xC | 40.2% | 77.2% | 51.5% | 24.0% | 77.0% | 83.9% | 70.8% | 77.3% | **84.5%** | xC |
| skill-4 | 25.8% | xC | xC | 44.1% | 73.6% | 75.8% | 70.4% | 65.7% | 77.0% | 76.4% | **77.9%** | 76.0% |
| skill-5 | 26.8% | xW | 53.0% | 55.6% | xW | 57.0% | 62.1% | 39.1% | xW | 73.6% | 59.9% | **77.1%** |
