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

### EXP-004 — 分层 skill 注入消融(base / facts / +heuristics / +menu)  2026-06-20→21

**问题:** 前作只测"有/无 skill"二元。wiki 重构成 facts/menu/heuristics 三层后,测**每加一层知识的边际价值**。注入的是 forge 真·wiki 卡(`--skill-level`,逐字、仅去 `[[链接]]`/Cross-refs 噪音),不是手写摘要 → 测的是 wiki 工件本身。⚠️ **within-op**:卡是从 gemm_bf16_nt 自己蒸的,facts≈发 recipe,所以这测的是"知识打包/脚手架有没有用"而非**泛化**(泛化见后续 EXP-005 flash-attention)。

**设置:** DeepSeek `agent`,gemm_bf16_nt,max-attempts 12,n=5/臂,一卡一臂(计时干净)。4 臂 = `--skill-level {none,facts,heuristics,full}`(累积:facts ⊂ +heuristics ⊂ +menu;注入 ~0/3k/5.6k/7.3k token)。**agent 代码不变 = 同一把尺子。** 注:phase2(heuristics/full)首轮被 DeepSeek 从服务器间歇性断连全打挂(`APITimeoutError`,裸 curl 也 http=000),加 driver 层 wait-for-DeepSeek + retry-on-empty 重跑得干净 n=5(`experiments/exp004_phase2_waiter.sh`,不改 agent)。

**结果(best per rep, % of cuBLAS):**

| 臂 | 注入 | rep1–5 | 中位 | 均值 | 超 cuBLAS | 特征 |
|---|---|---|---|---|---|---|
| base | 无 | 68.5 / 29.2 / 30.0 / 22.6 / 40.5 | 30.0 | 38.2 | 0/5 | 裸能力,多 xC/xW |
| facts | 指令卡 | 98.0 / 78.7 / 93.6 / 94.4 / 92.3 | 93.6 | 91.4 | 0/5 | recipe 一上跳 ~3× |
| **heuristics** | +判断 | 97.0 / 46.1 / **100.2** / **100.0** / 98.7 | **98.7** | 88.4 | **2/5** | 最高天花板,1 个 dud |
| full | +菜单 | 95.1 / 95.8 / 90.8 / 91.2 / 90.1 | 91.2 | **92.6** | 0/5 | 最稳(无 dud),天花板低 |

**结论:**
1. **注入知识 ≫ base:** 任意一层都把 agent 从中位 ~30% 拉到 ~90%+(base 几乎不自主破张量核,0/5 过 70)。within-op 最硬的信号——但注意 facts 本身≈发答案,这是"脚手架有效",不是泛化。
2. **heuristics 解锁天花板:** 加"regime→technique 判断"(v 阶梯)后中位 98.7、**2 个 rep 超 cuBLAS(100.0/100.2)**——agent 偶尔重构出 swizzle+stmatrix 冠军路径。光给 facts(指令)够到 ~94 就上不去 → 判断类知识有独立增量。
3. **menu 拿天花板换稳定(非单调!):** 再加广度菜单(full),**峰值被压**(中位 98.7→91.2、无一超 cuBLAS),但**方差最小**(全 90–96、无 dud,均值 92.6 反而最高)。"上下文越多越好"是错的——未验证的广度卡稀释了对冠军路径的聚焦,代价换来少踩坑。
4. **打包甜点取决于目标:** 求峰值/超 cuBLAS → facts+heuristics(但有 dud 风险);求稳定 → full。n=5 偏小且 heuristics 均值被单个 dud(46%)拽低 → **中位比均值更能反映各臂典型水平**。

**产物:** `runs_exp004/{base,facts,heuristics,full}/gemm_bf16_nt_2026062*/`。

**逐 attempt 矩阵:**

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-1 | 18.7% | xC | 6.0% | xC | xC | xC | 17.6% | 18.6% | xC | 16.4% | xW | **68.5%** |
| base-2 | xC | 5.9% | xC | xW | 6.0% | xW | 28.4% | xC | **29.2%** | 20.3% | 26.3% | 28.9% |
| base-3 | 5.4% | xC | 25.4% | xC | 16.5% | 18.5% | xC | xC | xW | xC | **30.0%** | 18.8% |
| base-4 | xC | xW | xW | xW | xW | 0.8% | 3.0% | xW | 14.8% | xW | **22.6%** | 22.1% |
| base-5 | xW | 5.4% | xW | 17.4% | xW | xW | xC | 13.4% | 17.3% | **40.5%** | . | . |
| facts-1 | xC | xC | 83.9% | **98.0%** | 93.7% | 77.6% | 78.3% | 96.3% | 66.8% | 93.5% | 58.3% | 61.8% |
| facts-2 | xW | 51.5% | 56.0% | xW | xW | 78.4% | xW | xW | 76.4% | **78.7%** | xW | 77.9% |
| facts-3 | 78.8% | 81.8% | 92.4% | 24.5% | xW | 24.5% | 74.3% | 24.6% | 81.9% | 93.6% | **93.6%** | 92.4% |
| facts-4 | 42.3% | 52.2% | xC | 77.3% | 75.4% | xC | xC | 93.4% | 77.3% | xW | **94.4%** | 92.2% |
| facts-5 | 49.2% | 73.5% | 70.5% | 73.3% | 67.2% | 72.6% | 72.6% | xW | **92.3%** | 89.1% | . | . |
| heuristics-1 | 50.8% | xC | 34.7% | 49.4% | xC | 46.1% | 51.1% | 91.0% | 84.0% | 96.6% | 84.0% | **97.0%** |
| heuristics-2 | xW | xW | xW | xC | xC | 5.9% | xW | 4.8% | 17.3% | 19.4% | 19.4% | **46.1%** |
| heuristics-3 | 94.2% | xC | xC | 94.6% | xC | xC | 93.1% | 93.9% | 96.8% | 90.4% | **100.2%** | . |
| heuristics-4 | 90.9% | xC | xC | 93.3% | 93.7% | 73.1% | 95.7% | xC | 96.8% | 99.5% | 90.5% | **100.0%** |
| heuristics-5 | xC | 94.5% | 94.8% | 97.2% | 97.3% | 97.2% | 92.0% | 96.5% | 95.1% | xC | 97.8% | **98.7%** |
| full-1 | 50.8% | 50.8% | 50.8% | 88.5% | 92.1% | xC | xW | 91.6% | 92.3% | xC | **95.1%** | 33.8% |
| full-2 | 95.6% | 69.0% | xW | 95.7% | xC | xC | **95.8%** | 94.0% | xC | 95.7% | 20.4% | xC |
| full-3 | 84.3% | xW | xW | 84.2% | 87.9% | 88.1% | xC | 88.4% | **90.8%** | 74.7% | 90.0% | 90.5% |
| full-4 | 88.4% | xC | xC | xW | **91.2%** | 63.6% | xW | 90.1% | 19.4% | xW | 90.8% | 91.1% |
| full-5 | xC | **90.1%** | 88.9% | 86.0% | 18.5% | xW | 17.9% | 89.4% | 18.2% | 87.5% | 85.5% | 90.1% |

### EXP-005 — 泛化测试:gemm 蒸的卡能不能帮到 flash_attention  2026-06-21

**问题:** EXP-004 是 within-op(卡从 gemm 蒸、注回 gemm,facts≈发答案)。真正的论文 claim 是**跨算子泛化**:把 **gemm** 的卡注给一个**没见过的、不同算法**的 agent,还灵吗?op = `flash_attention_bf16_fwd_bhsd`(算法不同:QKᵀ→online softmax→PV;但内部两个 matmul 用的是同一套张量核原语 mma/ldmatrix/cp.async/swizzle)。注入的是**gemm 的 wiki 卡原样**(facts=gemm 指令卡)——通用原语该迁移、gemm 专属 heuristics 不该迁移。

**设置:** DeepSeek agent,max-attempts 12,n=5/臂,一卡一臂。3 臂 = base / +gemm-facts / +gemm-heuristics(同 EXP-004 的 `--skill-level`,注的还是 gemm 卡)。**FA 正确性按 fp32 真值判**(`sampled_vs_fp32_math_allclose`)而非 cuDNN-bitmatch:FA 的 `allclose_vs_cudnn` atol=0.002 < 1 bf16 ULP(0.0156),会误杀正确 kernel(causal 大幅值输出和 cuDNN 因**分块加法顺序不同**差 1 个 bf16 取整位)——这是 draft op 的容差 bug,已记反馈团队。速度仍对 cuDNN 计。floor = forge `v1_naive` = 1.6%(fp32 门下五形状全对)。

**结果(best per rep, % of cuDNN):**

| 臂 | 注入 | rep1–5 | 中位 | 均值 |
|---|---|---|---|---|
| base | 无 | 4.2 / 9.7 / 3.0 / 3.9 / 2.7 | 3.9 | 4.7 |
| **facts** | gemm 指令卡 | 9.5 / 7.7 / 2.0 / 10.8 / 4.2 | **7.7** | 6.8 |
| heuristics | +gemm 阶梯 | 7.1 / 9.0 / 6.5 / 2.0 / 9.3 | 7.1 | 6.8 |

**结论:**
1. **泛化是真的,但弱:** gemm-facts 把 base 从 ~4% 抬到 ~7%(**~1.7×**)。mma/ldmatrix/swizzle 这些**通用 PTX 原语确实跨算子迁移**——agent 拿去把 FA 的两个 matmul 写快了点。
2. **跨算子 vs 同算子 = 1.7× vs 3×,直接量化了"定制教程"成分:** EXP-004 facts 让 gemm 30%→94%(~3×),这里 FA 4%→7%(~1.7×)。**EXP-004 那个暴涨的大头是 within-op 过拟合(发答案),真正可迁移的通用 skill 只值 ~1.7× 残差。** 这是对"wiki 是不是给这题定制的教程"最直接的数据回答:大部分是,小部分(原语)真泛化。
3. **gemm heuristics 不迁移(如预期):** heuristics ≈ facts(中位 7.1 vs 7.7,均值并列 6.8)——gemm 专属的 tile/pipeline 判断对 FA 既没加成也没明显误导。
4. **三臂全 <11% cuDNN:** cuDNN 融合 FA 太强;gemm 卡只含 matmul 原语,不含 FA 特有的 online-softmax/融合/tiling,所以迁移**有限**。要更强的 FA,得有 FA 自己的卡(future:写张量核 FA v2 + 蒸 FA 卡)。
5. **⚠️ 强度:** 数值个位数 %、n=5 方差大、xC/xW 多 → 方向(facts>base、heuristics≈facts)中位与均值一致,属**提示性**,不是强统计。

**产物:** `runs_exp005/{base,facts,heuristics}/flash_attention_bf16_fwd_bhsd_2026062*/`。

**逐 attempt 矩阵:**

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-1 | xC | 1.7% | xC | 3.6% | **4.2%** | xC | xW | 0.7% | 0.9% | 1.6% | 0.7% | 4.1% |
| base-2 | 6.5% | xW | xW | xW | xW | xW | 6.5% | 8.6% | **9.7%** | 8.3% | xC | 8.4% |
| base-3 | xC | xW | xW | xW | xW | **3.0%** | 2.0% | xW | 2.9% | xW | 1.4% | 2.0% |
| base-4 | xC | xW | 1.3% | xC | 3.5% | **3.9%** | xC | 3.4% | 3.0% | xC | 3.7% | xW |
| base-5 | 1.8% | xC | **2.7%** | . | . | . | . | . | . | . | . | . |
| facts-1 | 0.4% | xC | xC | 0.6% | xW | **9.5%** | xW | xW | 1.4% | xC | 6.7% | xW |
| facts-2 | 0.5% | xW | xC | 2.4% | 2.3% | xW | xC | 2.8% | 0.6% | **7.7%** | xC | 5.5% |
| facts-3 | xC | xW | xW | xW | 0.9% | xW | xW | xW | 1.1% | **2.0%** | 1.9% | 1.7% |
| facts-4 | xC | xW | 1.9% | xW | 1.8% | xC | xC | xW | xC | xC | 2.0% | **10.8%** |
| facts-5 | xW | 3.6% | xC | 3.9% | xW | xW | xW | xW | xW | 3.8% | xW | **4.2%** |
| heuristics-1 | xC | xC | xW | 5.7% | 5.6% | xW | **7.1%** | xW | 4.8% | 5.4% | 4.9% | xC |
| heuristics-2 | xC | 4.0% | 3.3% | 4.6% | xC | 3.5% | 3.2% | 5.1% | 6.0% | 6.5% | **9.0%** | 6.6% |
| heuristics-3 | xC | xW | xW | **6.5%** | . | . | . | . | . | . | . | . |
| heuristics-4 | xC | xC | xC | **2.0%** | xC | . | . | . | . | . | . | . |
| heuristics-5 | 3.2% | 5.5% | 5.5% | xW | xW | xW | 8.0% | **9.3%** | xW | xW | 4.4% | xW |

### EXP-006 — op 专属卡的边际价值:FA 自己的卡能不能强于通用指令卡  2026-06-22→23

**问题:** EXP-005 发现 gemm 卡注 FA 只弱迁移(~1.7×),因为只含通用 matmul 原语、没 FA 特有知识。那 FA **自己那张卡**(`facts/flash-attention-forward.md`:online-softmax 结构 + 张量核 QKᵀ/PV 映射 + cuDNN 容差坑)在通用原语之上还能再加多少?这是关掉泛化三部曲的最后一块。op 同 EXP-005。

**设置:** DeepSeek agent,max-attempts 12,n=5/臂,FA 正确性按 fp32 真值判(同 EXP-005)。为干净隔离 FA 卡,给 `wiki.load_bundle` 加了 `exclude={card}` + CLI `--wiki-exclude`(见 anvil dev `5f827b8`)。3 臂:
- **base** = 无注入
- **instr** = `facts` 去掉 FA 卡(只剩 mma/ldmatrix/cp.async/stmatrix/swizzle 这些**跨算子可迁移的指令原语**——等于在当前 DeepSeek 条件下**同期重跑** EXP-005 的 facts 条件,因 EXP-005 时 FA 卡还不存在)
- **instrfa** = 全 `facts`(instr + FA 专属卡)

headline = `instrfa − instr` = **对的算子自己那张卡**的边际价值。

**结果(best per rep, % of cuDNN):**

| 臂 | 注入 | rep1–5 | 中位 | 均值 | 峰值 |
|---|---|---|---|---|---|
| base | 无 | 7.4 / 1.4 / 3.8 / 8.0 / 10.1 | 7.4 | 6.1 | 10.1 |
| instr | 通用指令卡(无FA卡) | 5.4 / 12.3 / 4.2 / 3.1 / 21.2 | 5.4 | 9.2 | 21.2 |
| **instrfa** | +FA 专属卡 | 42.9 / 1.2 / 18.2 / 25.2 / 7.1 | **18.2** | 18.9 | **42.9** |

> instrfa rep5(7.1%)是补跑的:原 rep5 跑到第 3 个 attempt 时 DeepSeek 余额耗尽(402),被截断成 0.69% 的废数据,充值后单独补满 12 attempt 重跑;截断版留底 `*.balance-truncated.bak`,未删。

**结论:**
1. **对的算子自己的卡 = 强迁移:** instrfa 中位 18.2% vs instr 5.4% = **~2.5×**;vs base 7.4% = ~2.4×。峰值 42.9%(逼近 forge 手写 v7 的 56% 一半多、超过手写 v5 的 32%)。**FA 卡那 ~3.9KB 文本,值一个数量级的性能。**
2. **泛化三部曲闭环:** EXP-004 within-op facts ≈3×(过拟合发答案)/ EXP-005 gemm→FA 指令卡 ≈1.7×(弱、只迁原语)/ EXP-006 FA→FA 专属卡 ≈2.5×(强)。**一句话:卡能不能强迁移,看它是不是"对的算子"的卡;跨算子只有通用指令原语漏得过去。**
3. **instr ≈ base(中位 5.4 vs 7.4),复现 EXP-005:** 通用指令卡**没稳定抬中位**,只把尾巴拉肥(峰值 21.2% vs base 10.1%)——偶尔能让 agent 写出张量核冲一把,但不可靠。
4. **⚠️ 强度:** n=5、方差极大(instrfa 1.2↔42.9),中位/均值同向 → 提示性强信号,非强统计;但 2.5× 的差距远大于 EXP-005 的噪声尺度。

**迭代动态发现(回答"agent 很看第一版"):** 看每个 rep 的 **best 出现在第几 attempt**:

| 臂 | best@attempt(/12) | ncorrect/rep |
|---|---|---|
| base | 12, 12, 12, 11, 8 | 6,5,4,7,10 |
| instr | 6, 11, 10, 7, 12 | 5,4,3,1,7 |
| instrfa | 10, 11, 12, 12, 12 | 8,2,4,6,8 |

- best **几乎都在 attempt 10–12**(预算边缘),attempt 1 几乎都 fail/极低 → best **不是第一版**,它**在爬**。
- 但爬的是**单一结构内部**:rep 间 1.2↔42.9 的天差地别来自**第一个 correct kernel 选了标量还是张量核**(标量天花板 ~5–7%,张量核 ~30–42%),不是来自某 rep 爬得好。
- prompt 在**强化锁定**:`agent_feedback` 的 revert-to-best 说「回到最好那版**只改一处**」+「correctness first」——利于**局部爬坡**,但**压制了"推倒换结构"那一跳**(标量→张量核是整篇重写)。
- ⇒ agent 是 **hill-climber 非 basin-hopper**;**"很看第一版"准确说是"很看第一个 correct kernel 的路线"**。这也解释了 instrfa 为何强:**FA 卡让第一版就落在张量核高盆地** → 卡的主要价值 = 选对起跑盆地,而非教爬坡。
- "轮次够不够":best 贴预算边缘 → 当前盆地没爬满,多轮能再榨一点;但多轮的「只改一处」跳不出盆地,**主因是盆地锁定不是轮次**。**待验:** 挑一臂 `--max-attempts 24` 跑几 rep——best 仍贴边缘=轮次限;早早饱和=盆地限(则要改 prompt 逼换结构,而非加轮次)。

**逐 attempt 矩阵**(xC=编译/启动失败 · xW=算错 · %=正确的 geomean · **粗体**=该 rep best · `.`=没跑到该轮):

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-1 | xC | 1.1% | xW | xW | 0.9% | 1.7% | xW | xW | 2.1% | 6.7% | xW | **7.4%** |
| base-2 | 1.3% | xW | xW | 1.4% | 1.3% | xW | 1.2% | xW | xW | xW | xC | **1.4%** |
| base-3 | xC | xW | 1.9% | xW | 2.6% | xW | xW | xW | xW | xW | 3.7% | **3.8%** |
| base-4 | xW | xW | 6.7% | 5.0% | 7.9% | 7.3% | 7.6% | xW | 8.0% | xW | **8.0%** | xW |
| base-5 | 4.8% | xC | 1.8% | 8.4% | 8.2% | 5.2% | xW | **10.1%** | 9.5% | 9.2% | 3.0% | 9.6% |
| instr-1 | 2.3% | xW | xW | 4.9% | 4.4% | **5.4%** | xW | 2.7% | . | . | . | . |
| instr-2 | xC | 5.0% | xC | xW | xC | 7.7% | 7.0% | xW | xC | xC | **12.3%** | xC |
| instr-3 | xC | xC | xW | xW | xW | 1.6% | xW | xW | 3.5% | **4.2%** | xW | xW |
| instr-4 | xW | xW | xW | xW | xW | xW | **3.1%** | . | . | . | . | . |
| instr-5 | 1.4% | xC | 1.4% | xW | 9.8% | xC | 9.9% | xC | 14.8% | xW | 15.7% | **21.2%** |
| instrfa-1 | xC | 9.7% | 23.2% | 23.0% | 22.8% | xC | 26.5% | 42.4% | 39.3% | **42.9%** | xC | xC |
| instrfa-2 | xC | xC | xC | xC | xC | xC | xW | xW | 1.1% | xW | **1.2%** | xC |
| instrfa-3 | xC | xW | xW | xC | xC | 4.2% | xC | xC | 7.9% | 14.0% | xW | **18.2%** |
| instrfa-4 | xW | xW | 1.1% | xC | xW | 2.0% | xW | 0.9% | xC | 2.4% | 13.2% | **25.2%** |
| instrfa-5 | 6.2% | 2.7% | 6.2% | xC | 6.7% | xC | xW | 1.1% | 6.6% | xW | 6.4% | **7.1%** |

读矩阵即见盆地锁定:`instrfa-1` 一旦在 attempt 2 进了张量核盆地就稳步爬(9.7→23→26→42→**42.9%**);而 `base-*`/`instr-*` 多数 rep 在 ≤10% 反复横跳、满屏 xW——从没跳进高盆地。`instrfa-2` 则是反例(开局连环 xC 没立住结构,卡死 1.2%)。

**产物:** `runs_exp006/{base,instr,instrfa}/flash_attention_bf16_fwd_bhsd_2026062*/`;逐 attempt 明细在各 run 的 `results.jsonl`。
