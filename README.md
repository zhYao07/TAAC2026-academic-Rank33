# TAAC 2026 / KDD Cup 2026 Tencent UniRec Challenge Solution

本仓库开源了我们在 **TAAC 2026** 初赛和复赛最高分两份代码。方案以官方 **HyFormer** baseline 为基础，重点围绕数据理解、特征处理、tokenization、目标感知序列建模和训练稳定性进行改进。
> 初赛最终方案将公开榜 AUC 从官方 baseline 的 **0.809** 提升至 **0.83269**。  
> 复赛最终方案将公开榜 AUC 从官方 baseline 的 **0.820922** 提升至 **0.833983**。  

## 目录结构

```text
.
├── first_round/               # 初赛最佳代码
│   ├── dataset.py             # Parquet 数据读取、时间与 dense 特征处理
│   ├── model.py               # HyFormer、DIN、LongerEncoder 等模型组件
│   ├── trainer.py             # 训练、EMA、SWA 与 checkpoint 管理
│   ├── train.py               # 训练入口
│   ├── infer.py               # 推理入口
│   ├── run.sh                 # 初赛最佳配置
│   └── ns_groups.json
├── second_round/              # 复赛最佳代码
│   ├── dataset.py             # 数据读取与特征构造
│   ├── item_feature_engineering.py
│   ├── seq_word2vec.py        # 高基数字段的紧凑 word2vec 恢复
│   ├── model.py               # 复赛最终模型
│   ├── trainer.py             # DDP、EMA、SupCon 与 checkpoint 管理
│   ├── train.py
│   ├── infer.py
│   ├── run.sh                 # 复赛最终配置
│   └── ns_groups.json
└── KDD_Cup_V2/                # 方案论文及配图
```

## 方法概览

我们没有大幅重写官方 HyFormer backbone，而是优先提升输入 token 的信息质量，并在 query generator 中加入轻量的候选感知能力。

主要改进包括：

- **尺度感知的 dense tokenization**：按特征分布和语义对 dense 字段分组，对不同分支分别使用 raw、`log1p` 或 signed `log1p` 变换，避免异构数值被压进同一个 token。
- **时间与日历特征**：使用样本级时间特征、序列级 time bucket，以及周末、节假日等 day-type 信息。
- **目标感知 query generator**：使用 DIN 风格的 candidate-aware pooling，让每个序列域的 query 更关注与当前候选 item 相关的历史行为。
- **多时间尺度序列建模**：针对不同序列域的长度与时间跨度采用不同处理方式；初赛中对密集的 `seq_d` 使用 LongerEncoder。
- **item 历史统计**：从曝光侧构造轻量 item 描述符，不使用评估标签。
- **高基数信号恢复**：先以 hash embedding 验证信号有效性，最终使用 compact frozen word2vec 恢复高填充率、高基数的序列与非序列字段。
- **训练稳定化**：使用 dense-only EMA、正样本作为 anchor 的监督对比学习，以及全量数据训练。

## 初赛上分记录

以下是初赛阶段的累计实验轨迹，每一项均建立在前一版本之上：

| 阶段                            |         AUC |
| ------------------------------- | ----------: |
| Baseline                        |      0.8090 |
| 加入时间特征                    |      0.8240 |
| 重新处理 dense 特征             |      0.8270 |
| Cross Attention 替换为 DIN      |      0.8278 |
| Sequence token 标注节假日、周末 |      0.8287 |
| 加入 EMA                        |      0.8289 |
| 加入 `torch.compile`            |      0.8292 |
| `seq_d` 改为 LongerEncoder      |      0.8304 |
| DIN 中加入时间信息              |      0.8311 |
| 加入 SWA                        |      0.8319 |
| Item token 加入时间特征         |      0.8323 |
| 调整 SWA 融合权重               | **0.83269** |

初赛最终代码位于 [`first_round/`](first_round/)。其中 SWA 会保存最佳、次优和融合后的自包含 checkpoint；当前融合权重为最佳模型 `0.6`、次优模型 `0.4`。

## 复赛上分记录

复赛实验以官方公开榜 AUC 为指标。下表来自论文中的累计迭代日志，因此应理解为工程优化轨迹，而不是多随机种子、严格控制变量的消融实验。

| 阶段                         |          AUC |  相对提升 |
| ---------------------------- | -----------: | --------: |
| 官方 baseline                |     0.820922 |         — |
| Refined dense token          |     0.828658 | +0.007736 |
| 加入 EMA                     |     0.831523 | +0.002865 |
| Hash-based 高基数恢复        |     0.831962 | +0.000439 |
| 日历与时间特征               |     0.832166 | +0.000204 |
| Item 历史统计                |     0.832438 | +0.000272 |
| Positive-anchor SupCon       |     0.832669 | +0.000231 |
| Target-aware query generator |     0.832764 | +0.000095 |
| 两个 item dense token        |     0.833074 | +0.000310 |
| 全量数据训练                 |     0.833277 | +0.000203 |
| 七个序列字段的 word2vec      |     0.833607 | +0.000330 |
| 最终 compact frozen word2vec |     0.833768 | +0.000161 |
| 模型 scaling                 | **0.833983** | +0.000215 |

最终 scaling 配置为：

| 参数                      | 配置                                               |
| ------------------------- | -------------------------------------------------- |
| `d_model` / embedding dim | 108 / 64                                           |
| HyFormer                  | 2 blocks，6 heads                                  |
| FFN multiplier            | 6                                                  |
| Query / sequence budget   | 每个序列域 2 queries，top-50 tokens                |
| Sequence encoder          | SwiGLU                                             |
| Batch size / epochs       | 1024 / 8                                           |
| EMA                       | dense-only，decay 0.9995                           |
| SupCon                    | weight 0.1，temperature 0.1，positive anchors only |
| 高基数恢复                | compact frozen word2vec                            |

## 环境

建议使用 Linux、Python 3.9+ 和支持 CUDA 的 PyTorch 环境。核心依赖如下：

```bash
pip install torch numpy pyarrow pandas scikit-learn tqdm
```

复赛训练会在任务内构建 word2vec，因此还需要：

```bash
pip install gensim
```

实际可用的 batch size、worker 数量和 word2vec 词表规模取决于 CPU 内存、GPU 显存及训练时间预算。复赛默认配置面向多 GPU 环境，并在检测到多张 GPU 时自动通过 `torchrun` 启动 DDP。

## 数据准备

受比赛数据许可约束，本仓库不包含原始数据。训练目录需要包含官方提供的 Parquet 文件及 `schema.json`：

```text
train_data/
├── schema.json
├── part-00000.parquet
├── part-00001.parquet
└── ...
```

评估目录同样应包含测试 Parquet 文件；如果模型目录中没有 `schema.json`，推理脚本会回退使用评估目录下的 schema。

## 训练

训练入口支持命令行参数，也支持比赛环境变量。环境变量优先于命令行参数：

```bash
export TRAIN_DATA_PATH=/path/to/train_data
export TRAIN_CKPT_PATH=/path/to/checkpoints
export TRAIN_LOG_PATH=/path/to/logs
```

运行初赛最佳配置：

```bash
bash first_round/run.sh
```

运行复赛最佳配置：

```bash
bash second_round/run.sh
```

也可以直接调用训练脚本并覆盖默认参数：

```bash
python first_round/train.py \
  --data_dir /path/to/train_data \
  --ckpt_dir /path/to/checkpoints \
  --log_dir /path/to/logs
```

```bash
python second_round/train.py \
  --data_dir /path/to/train_data \
  --ckpt_dir /path/to/checkpoints \
  --log_dir /path/to/logs
```

复赛 `run.sh` 默认使用全部训练数据（`valid_ratio=0`），训练 8 个 epoch，并为每个 epoch 保存自包含 checkpoint。若希望在本地选择最佳模型，可覆盖为非零验证比例，例如：

```bash
bash second_round/run.sh --valid_ratio 0.1
```

## 推理

训练生成的 checkpoint 子目录应至少包含：

```text
checkpoint_dir/
├── model.pt
├── schema.json
└── train_config.json
```

如果启用了相应配置，word2vec 的紧凑词表、raw-ID lookup 和 embedding 参数已随 `model.pt` 保存，推理时不需要额外的 word2vec 文件。

设置推理环境变量：

```bash
export MODEL_OUTPUT_PATH=/path/to/checkpoint_dir
export EVAL_DATA_PATH=/path/to/eval_data
export EVAL_RESULT_PATH=/path/to/result
```

初赛模型：

```bash
python first_round/infer.py
```

复赛模型：

```bash
python second_round/infer.py
```

预测结果将写入：

```text
/path/to/result/predictions.json
```

格式为：

```json
{
  "predictions": {
    "user_id_1": 0.123,
    "user_id_2": 0.456
  }
}
```

## 复现说明

- 排行榜结果来自累计提交记录，小幅提升可能受到公开榜划分、提交顺序和随机性的影响。
- 复赛最终全量训练没有内部验证集，需要根据实验记录选择提交 epoch；论文 scaling 实验中的最佳 checkpoint 出现在 epoch 5。
- `item_feature_engineering.py` 构造的是比赛侧静态曝光统计。若用于在线系统，需要按时间因果地计算特征，并处理延迟反馈。
- 初赛与复赛的 `model.pt` 结构并不通用，请始终使用对应目录中的 `infer.py` 加载。
- `schema.json`、`train_config.json` 和模型权重必须来自同一次训练，以保证严格加载成功。

## 低收益或无收益尝试

复赛中我们还尝试了 focal loss、label smoothing、BatchLogitNCE、FGM、SWA top-2 fusion、time-based validation、显式 user-item pair token、item-ID cold-start bucketing、OOF item 标签统计、更深的 HyFormer，以及更重的 target-aware decoding 等方向。这些尝试整体收益较低或出现掉分。

我们的主要经验是：**官方 HyFormer 已经是很强的统一 backbone；相比大规模重写结构，从 EDA 出发恢复被忽略的信号、构造更稳定的 token，通常更加有效。**

## 致谢

感谢 KDD Cup 2026 Tencent UniRec Challenge / TAAC 2026 组织方提供任务、数据、评测平台和官方 HyFormer baseline。

如在研究中使用本仓库，请同时引用原始 HyFormer 工作及本仓库中的方案论文。

## 开源前检查

发布前请根据你的开源计划补充合适的 `LICENSE` 文件，并确认比赛数据、官方代码及模型权重的再分发条款。本仓库的代码开源不代表比赛数据可以公开传播。
