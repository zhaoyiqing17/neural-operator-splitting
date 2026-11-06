# NOS 字典约简实验代码包说明

本代码包是 NOS（Serrano 等，arXiv:2602.00884）字典冗余约简实验的新增评测代码。
解压后覆盖到官方仓库根目录 `/code/neural-operator-splitting`，全部为**新增文件**，
不修改官方仓库已有文件。

## 文件清单与作用

| 文件 | 作用 |
|---|---|
| `test_time_compute/reduction.py` | 三种约简策略 + 子集抽取：`random_reduce`（随机不放回；seed=42 即论文 C.5 口径）、`frequency_reduce`、`cluster_reduce` |
| `test_time_compute/reduce_dictionary.py` | 主运行入口（平流-扩散）：构建完整字典 → 约简 → 测试集束搜索 → 落指标 |
| `test_time_compute/metrics.py` | 指标计算与 CSV/JSON 落盘：退化%、候选数、参数 MAE、random 均值±标准差 |
| `test_time_compute/analysis/plot_reduction.py` | 画 N′–退化% 与 N′–候选数 两条曲线 |
| `setup_nos.sh` | 复用 `/data/nestor_env/envs/nestor` 并补装依赖，输出 `environment check: PASS` |
| `run_reduce.sh` | 统一入口：`--check-only` / `baseline` / `smoke` / `full` 四种模式 |
| `monitor_nos.sh` | 查看运行日志与 `summary.json` |

## 快速上手

```bash
cd /code/neural-operator-splitting
export ENV_DIR=/data/nestor_env/envs/nestor
export PATH="$ENV_DIR/bin:$PATH"

bash setup_nos.sh                     # 安装/复用环境
bash run_reduce.sh --check-only       # 预检
bash run_reduce.sh baseline           # 复现完整字典基线 (≈0.052 / ≈0.002)
bash run_reduce.sh smoke              # 30 条轨迹冒烟 (N'=64)
nohup bash run_reduce.sh full > /data/nos_work/runs/dict_reduce/console.log 2>&1 &
```

## 复现论文 C.5 / Table 7（可选，验证 harness）

论文附录 C.5 的 Table 7 是在「平流+扩散合成」任务（`E_AD_ALL`，v,D∈[0.01,1.0]）上、
随机不放回子采样 seed=42 得到的。用下面命令逐点复现，确认环境/搜索配置与论文一致：

```bash
python test_time_compute/reduce_dictionary.py \
  --model_path /model/nos_disco/advection-diffusion/last.ckpt \
  --tasks E_AD_ALL --strategies random \
  --n_operators 16 32 64 256 --random_seeds 42 \
  --output_dir /data/nos_work/results/dict_reduce/c5
```

预期（论文 Table 7）：N=16→0.037，N=32→0.035，N=64→0.017，N=256→0.015。
注意：这是合成任务，不是本实验主测的 E_AD_v / E_AD_D 外推任务。

## 命令行参数（reduce_dictionary.py）

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model_path` | checkpoint 路径 | 必填 |
| `--tasks` | `E_AD_v`（c∈[1,3]）、`E_AD_D`（D∈[1,3]）、`E_AD_ALL`（合成，C.5 复现） | `E_AD_v` |
| `--strategies` | `full/random/frequency/cluster` | 全部 |
| `--n_operators` | 目标规模 N′ | 16 32 64 128 |
| `--random_seeds` | random 策略的种子 | 0 1 2 |
| `--beam_width / --max_operators / --improvement_threshold` | 束搜索 B / M / 5% | 4 / 5 / 5.0 |
| `--n_dict_batches` | 完整字典 = 该值×64 | 4（=256） |
| `--heldout_batches` | 留出训练轨迹 = 该值×64 | 2（=128） |
| `--heldout_seed` | 留出集生成种子 | 7 |
| `--num_samples` | 每任务测试样本数 | 512 |
| `--pca_dim` | 聚类 PCA 维数 | min(32, N′, d) |

## 输出字段（history.csv 每行）

`task`、`strategy`、`n_operators`、`seed`、`nrmse`、`degradation_pct`（相对完整字典）、
`candidates`（B×N′×M 上界）、`wall_time_s`、`param_mae_c`、`param_mae_D`、
`traceable`（cluster 为 False）。`summary.json` 额外含 random 的 `nrmse_std` 与逐种子结果。

## 复现种子清单

- 字典生成：`TemporalBatchDatasetFly(seed=42)`
- 测试集：`TemporalBatchDatasetFly(seed=124)`，测试 power=3
- 留出集（frequency 计数 + cluster 权重）：`seed=7`
- random 策略：{0, 1, 2} 用于均值±标准差；seed=42 即论文 `sweep_ad_subsampling.py` 的 C.5 口径
- 全局 torch/numpy 种子：1234

## 组合方程（Step 5，可选）

`reduction.py` 的策略与数据无关，可直接复用于组合方程。接入时用
`test_time_compute/equations/test_combined_equation.py` 的 `CombinedHDF5TemporalDataset`
与 `train.train_combined_aggregate.DISCOLitModule` 构建字典（标签改为
`alpha/beta/gamma`），并在束搜索前插入同样的约简步骤。此适配在 Step 1–4 结果确认后再生成。
