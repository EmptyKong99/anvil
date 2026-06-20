# 逐 attempt 结果矩阵

> 自动生成:`python collect_results.py > RESULTS.md`(读各 run 的 results.jsonl)。
> 行=step(Route-B iter / agent attempt),列=run。EXPERIMENTS.md 是叙事+汇总,这里是明细。

> 格子:数字=加速比(geomean vs cuBLAS,正确) · `xC`=编译/校验失败 · `xW`=结果错 · `.`=该步未跑。列名 b=base(no-skill)/s=skill,后接 rep 号。

### exp001
| step | b1 | s1 |
|---|---|---|
| 1 | xC | xC |
| 2 | xW | xC |
| 3 | xC | xW |
| 4 | xC | xC |
| 5 | 0.031 | 0.577 |
| 6 | xC | xC |

### exp002
| step | b1 | b2 | b3 | s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| 1 | xC | 0.113 | xC | 0.327 | 0.343 | xC |
| 2 | xW | xC | xC | 0.563 | xC | xC |
| 3 | 0.204 | xW | 0.058 | xW | 0.759 | xC |
| 4 | xC | xC | xC | 0.568 | 0.711 | 0.050 |
| 5 | 0.783 | xW | 0.060 | 0.545 | 0.741 | xW |
| 6 | 0.618 | xW | xC | 0.648 | 0.880 | xC |
| 7 | 0.464 | xC | xC | 0.680 | 0.823 | xW |
| 8 | 0.311 | xC | xC | 0.716 | xC | xW |
| 9 | 0.108 | 0.113 | 0.059 | xW | 0.881 | xC |
| 10 | 0.319 | xW | xC | 0.573 | 0.883 | 0.060 |
| 11 | xC | 0.060 | xC | 0.486 | 0.825 | xC |
| 12 | 0.568 | xC | xC | 0.746 | 0.169 | xW |

