# TAAC 2026 / KDD Cup 2026 Tencent UniRec Challenge Solution

本仓库开源了我们在 **TAAC 2026** 初赛和复赛最高分两份代码。方案以官方 **HyFormer** baseline 为基础，重点围绕数据理解、特征处理、tokenization、目标感知序列建模进行改进。
> 初赛最终方案将公开榜 AUC 从官方 baseline 的 **0.809** 提升至 **0.83269**。  
> 复赛最终方案将公开榜 AUC 从官方 baseline 的 **0.820922** 提升至 **0.833641**，比赛结束后开放的一周评测机会后，我们做了一些scaling实验将分数提升到了0.833983
<img width="1232" height="213" alt="image" src="https://github.com/user-attachments/assets/35414609-995c-49b5-a328-b6409df23494" />


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

```


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

复赛实验以官方公开榜 AUC 为指标。

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
| Query / sequence budget   | 每个序列域 2 queries               |
| Sequence encoder          | SwiGLU                                             |
| Batch size / epochs       | 1024 / 5                                          |
| EMA                       | dense-only，decay 0.9995                           |
| SupCon                    | weight 0.1，temperature 0.1，positive anchors only |
| 高基数恢复                | compact frozen word2vec                            |



## 低收益或无收益尝试

初期我们将初赛中work的trick都移植到复赛中，发现大多都掉分或是提分很小。其中初赛上分最多的时间特征，在复赛中大概只有5个万分位的提升，其他比如Longer encoder，swa等等也都是掉分。复赛中我们还尝试了 focal loss、label smoothing、BatchLogitNCE、FGM、SWA top-2 fusion、time-based validation、显式 user-item pair token、item-ID cold-start bucketing、OOF item 标签统计、更深的 HyFormer，以及更重的 target-aware decoding 等方向。这些尝试整体收益较低或出现掉分。

我们的主要经验是：**官方 HyFormer 已经是很强的统一 backbone；相比大规模重写结构，从 EDA 出发恢复被忽略的信号、构造更稳定的 token，通常更加有效。**

## 致谢

感谢 KDD Cup 2026 Tencent UniRec Challenge / TAAC 2026 组织方提供任务、数据、评测平台和官方 HyFormer baseline。
