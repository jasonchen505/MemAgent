# MemAgent 8卡3090复现计划

> 资源：8×RTX 3090 (24GB显存)
> 目标：完整复现MemAgent训练流程，深入学习掌握项目

---

## 一、资源评估与可行性分析

### 1.1 显存需求分析

| 组件 | 原配置(A100 80GB) | 3090 (24GB)限制 |
|------|-------------------|-----------------|
| 模型参数(7B FP16) | ~14GB | 14GB ✓ |
| 优化器状态 | ~28GB | 需offload ✓ |
| 梯度 | ~14GB | 需gradient checkpointing ✓ |
| KV Cache(vLLM) | ~24GB | 需限制gpu_memory_utilization ✓ |
| 激活值 | ~10GB | 需gradient checkpointing ✓ |

### 1.2 可行性结论

**7B模型：✓ 可行，但需要激进优化**
- 必须开启：param_offload、optimizer_offload、gradient_checkpointing
- 减小：train_batch_size、rollout.n、ppo_max_token_len_per_gpu
- 预计训练时间：2-3周（原配置3-4天）

**14B模型：✗ 不可行**
- 14B FP16需要约28GB，超出3090单卡限制
- 即使offload，训练效率极低，不建议尝试

### 1.3 复现策略

```
Phase 1: 环境搭建 + 快速验证（1-2天）
Phase 2: 小规模训练调试（2-3天）
Phase 3: 完整训练复现（2-3周）
Phase 4: 评估与分析（2-3天）
```

---

## 二、Phase 1: 环境搭建与快速验证

### 2.1 环境安装

```bash
# Step 1: 创建conda环境
conda create -n memagent python=3.10 -y
conda activate memagent

# Step 2: 安装PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Step 3: 安装vLLM (关键依赖，必须0.8.2版本)
pip install vllm==0.8.2

# Step 4: 安装其他依赖
pip install -r requirements.txt

# Step 5: 安装flash-attn (可选，提升性能)
pip install flash-attn --no-build-isolation

# Step 6: 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```

### 2.2 数据准备

```bash
# Step 1: 下载QA数据集
cd taskutils/memory_data
bash download_qa_dataset.sh

# Step 2: 下载HotpotQA训练数据
cd ../..
bash hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10
export DATAROOT=$(pwd)/hotpotqa

# Step 3: 下载Qwen2.5-7B-Instruct模型
bash hfd.sh Qwen/Qwen2.5-7B-Instruct --tool aria2c -x 10
export MODELROOT=$(pwd)/models
mv Qwen2.5-7B-Instruct $MODELROOT/Qwen2.5-7B-Instruct-128K

# Step 4: 修改config.json激活YaRN（如果需要长上下文外推）
# 参考Qwen2.5-Instruct Repo的说明
```

### 2.3 快速验证：Quickstart

```bash
# 验证1: 使用预训练模型测试Memory机制
vllm serve BytedTsinghua-SIA/RL-MemoryAgent-7B --tensor_parallel_size 1

# 另一个终端
python quickstart.py --model BytedTsinghua-SIA/RL-MemoryAgent-7B --chunk 2000 --max_new 512

# 验证2: 测试基础Qwen模型（无RL训练）
python quickstart.py --model Qwen/Qwen2.5-7B-Instruct --chunk 2000 --max_new 512

# 对比两者输出质量，理解RL训练的效果
```

---

## 三、Phase 2: 小规模训练调试

### 3.1 创建适配3090的训练脚本

创建文件：`run_memory_7B_3090.sh`

```bash
#!/bin/bash
set -x

# 资源配置：1节点8卡3090
NNODES=1
NGPUS_PER_NODE=8

# 路径配置（需要修改）
PROJ_ROOT=/path/to/checkpoints
DATASET_ROOT=/path/to/hotpotqa
MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct-128K

VAL_PATH="${DATASET_ROOT}/hotpotqa_dev.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa_train_32k.parquet"
EXP=memory_agent/7B_3090
PROJ_DIR=${PROJ_ROOT}/${EXP}

MAXLEN=8192 
MAX_NEW_TOKEN=1024

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=16 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='memagent-3090' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=10
```

### 3.2 关键参数调整说明

| 参数 | 原值 | 3090值 | 调整原因 |
|------|------|--------|----------|
| `train_batch_size` | 128 | 16 | 减少单batch显存占用 |
| `rollout.n` | 16 | 4 | 减少生成数量，降低显存 |
| `ppo_mini_batch_size` | 8 | 2 | 减少mini-batch显存 |
| `ppo_max_token_len_per_gpu` | 16384 | 8192 | 减少每GPU token数 |
| `gpu_memory_utilization` | 0.6 | 0.4 | 为训练留更多显存 |
| `max_num_batched_tokens` | 16384 | 8192 | 减少vLLM批处理token数 |
| `total_epochs` | 30 | 10 | 先验证可行性 |
| `save_freq` | 10 | 5 | 更频繁保存，便于恢复 |

### 3.3 小规模调试运行

```bash
# Step 1: 先用更小的配置测试是否能跑通
# 创建一个极简测试脚本
cat > test_run.sh << 'EOF'
#!/bin/bash
python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=2000 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    ...
EOF

# Step 2: 监控显存使用
watch -n 1 nvidia-smi

# Step 3: 如果OOM，进一步减小配置
```

### 3.4 常见问题排查

**问题1: CUDA OOM**
```bash
# 解决方案：进一步减小以下参数
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
data.train_batch_size=4
```

**问题2: vLLM启动失败**
```bash
# 检查vLLM版本
pip show vllm
# 必须是0.8.2版本
pip install vllm==0.8.2
```

**问题3: Ray集群初始化失败**
```bash
# 清理Ray状态
ray stop
ray start --head
```

---

## 四、Phase 3: 完整训练复现

### 4.1 训练监控配置

```bash
# 在训练脚本中添加wandb配置
export WANDB_PROJECT=memagent-3090
export WANDB_NAME=7B-3090-run1
export WANDB_API_KEY=your_api_key

# 训练时会自动记录到wandb
trainer.logger=['console','wandb']
```

### 4.2 训练过程监控

**关键指标**：
```python
# 1. Reward变化
# 期望：从0逐渐上升到0.3-0.5（训练验证器）

# 2. KL散度
# 期望：保持在较低水平，不应持续上升

# 3. 生成长度
# 期望：Memory长度稳定在合理范围

# 4. 显存使用
# 期望：稳定在22-23GB，不应OOM
```

**监控命令**：
```bash
# 终端1：训练
bash run_memory_7B_3090.sh

# 终端2：显存监控
watch -n 1 nvidia-smi

# 终端3：日志监控
tail -f /path/to/logs/training.log
```

### 4.3 训练时间估算

| 阶段 | 原配置 | 3090配置 | 时间估算 |
|------|--------|----------|----------|
| 单步训练 | ~30s | ~3-5min | 10x慢 |
| 1 epoch | ~2h | ~20-30h | 10x慢 |
| 10 epochs | ~20h | ~200-300h | 8-12天 |
| 30 epochs(完整) | ~3天 | ~600-900h | 25-38天 |

**建议**：
- 先跑10 epochs验证效果
- 如果reward有明显上升，再决定是否继续
- 可以从checkpoint恢复训练

### 4.4 Checkpoint恢复训练

```bash
# 如果训练中断，从checkpoint恢复
python3 -m verl.trainer.main_ppo \
    ... \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$PROJ_DIR
```

---

## 五、Phase 4: 评估与分析

### 5.1 评估脚本准备

```bash
# 创建评估脚本
cat > eval_3090.py << 'EOF'
import os
import sys

# 使用训练好的模型进行评估
class Config:
    def __init__(self):
        self.name = "MemoryAgent-7B-3090"
        self.ckpt = "/path/to/checkpoints/memory_agent/7B_3090/global_step_XXX"
        self.tp = 1  # 3090单卡推理
        self.method = "recurrent"
        self.concur = 64  # 降低并发
        self.env = ENV(
            RECURRENT_MAX_CONTEXT_LEN=120000,
            RECURRENT_CHUNK_SIZE=5000,
            RECURRENT_MAX_NEW=1024
        )

# 只评估部分任务，节省时间
RULER_HQA_TESTS = [50, 100, 200, 400]  # 从短到长

def main():
    config = Config()
    # 评估代码...
EOF
```

### 5.2 评估任务选择

**由于资源有限，建议按优先级评估**：

| 优先级 | 任务 | 长度 | 目的 |
|--------|------|------|------|
| P0 | RULER_HQA | 50,100,200 | 验证基础能力 |
| P1 | RULER_HQA | 400,800 | 验证中等长度 |
| P2 | niah_single | 8K-32K | 验证OOD能力 |
| P3 | RULER_HQA | 1600+ | 验证长文本外推 |

### 5.3 结果分析

**与原论文对比**：
```python
# 原论文结果（参考）
original_results = {
    "7B_RULER_HQA_50": 0.95,
    "7B_RULER_HQA_100": 0.93,
    "7B_RULER_HQA_200": 0.90,
    # ...
}

# 你的结果
your_results = {
    "7B_3090_RULER_HQA_50": 0.XX,
    # ...
}

# 分析差异原因
# 1. 训练epoch数不足
# 2. batch_size减小导致训练不稳定
# 3. rollout.n减小导致探索不足
```

---

## 六、增量学习检查点

### 6.1 每Phase学习目标

**Phase 1 学习目标**：
- [ ] 理解verl框架的基本架构
- [ ] 理解vLLM的部署方式
- [ ] 理解quickstart.py的Memory机制
- [ ] 能运行quickstart并观察输出

**Phase 2 学习目标**：
- [ ] 理解训练脚本的每个参数含义
- [ ] 理解FSDP的显存优化原理
- [ ] 理解GRPO的实现细节
- [ ] 能成功运行小规模训练

**Phase 3 学习目标**：
- [ ] 理解训练过程中的指标变化
- [ ] 能定位和解决训练问题
- [ ] 理解checkpoint的保存和恢复
- [ ] 能监控训练状态

**Phase 4 学习目标**：
- [ ] 理解RULER benchmark的评估方式
- [ ] 能独立运行评估脚本
- [ ] 能分析训练效果
- [ ] 能对比不同配置的结果

### 6.2 关键代码阅读清单

| 阶段 | 必读文件 | 学习重点 |
|------|----------|----------|
| Phase 1 | `quickstart.py` | Memory机制的最简实现 |
| Phase 2 | `run_memory_7B_3090.sh` | 训练参数的含义 |
| Phase 2 | `recurrent/impls/memory.py` | MemoryAgent核心实现 |
| Phase 2 | `recurrent/interface.py` | RAgent接口设计 |
| Phase 3 | `recurrent/generation_manager.py` | 训练循环实现 |
| Phase 3 | `verl/workers/reward_manager/thread.py` | 分布式Reward计算 |
| Phase 4 | `taskutils/memory_eval/run.py` | 评估流程 |

---

## 七、快速参考

### 7.1 关键路径

```bash
# 项目根目录
MEMAGENT_ROOT=/home/chenyizhou/MemAgent

# 模型路径
MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct-128K

# 数据路径
DATAROOT=/path/to/hotpotqa

# Checkpoint路径
PROJ_ROOT=/path/to/checkpoints
```

### 7.2 常用命令

```bash
# 启动训练
bash run_memory_7B_3090.sh

# 监控显存
watch -n 1 nvidia-smi

# 启动Ray
ray start --head

# 停止Ray
ray stop

# 查看训练日志
tail -f /path/to/logs/*.log

# 恢复训练
python3 -m verl.trainer.main_ppo ... trainer.resume_mode=auto
```

### 7.3 紧急处理

**OOM处理**：
```bash
# 立即减小以下参数
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
data.train_batch_size=4
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
```

**训练卡住**：
```bash
# 检查Ray状态
ray status
# 重启Ray
ray stop && ray start --head
```

**vLLM报错**：
```bash
# 检查版本
pip show vllm
# 重装
pip install vllm==0.8.2 --force-reinstall
```

---

## 八、预期成果

### 8.1 最低目标（必须达成）

- [ ] 成功运行quickstart，理解Memory机制
- [ ] 成功运行小规模训练（1-2 epochs）
- [ ] 理解训练流程和关键参数
- [ ] 能独立排查常见问题

### 8.2 进阶目标（尽量达成）

- [ ] 完成10 epochs训练
- [ ] Reward有明显上升（>0.2）
- [ ] 完成部分RULER评估
- [ ] 与原论文结果有可比性

### 8.3 理想目标（资源允许）

- [ ] 完成30 epochs完整训练
- [ ] 评估结果接近原论文
- [ ] 能复现外推能力
- [ ] 能修改代码实现新功能

---

## 九、时间规划

| 天数 | 阶段 | 任务 | 产出 |
|------|------|------|------|
| Day 1-2 | Phase 1 | 环境搭建、数据准备 | quickstart可运行 |
| Day 3-5 | Phase 2 | 小规模训练调试 | 训练可跑通 |
| Day 6-25 | Phase 3 | 完整训练 | checkpoint |
| Day 26-28 | Phase 4 | 评估分析 | 评估结果 |

**总计：约4周**

---

## 十、学习资源

### 10.1 必读文档

1. `INTERVIEW_PREP.md` - 面试准备文档
2. `INTERVIEW_DEEP_DIVE.md` - 深度技术分析
3. verl官方文档：https://github.com/volcengine/verl
4. vLLM文档：https://docs.vllm.ai/

### 10.2 论文阅读

1. MemAgent论文：`paper/paper.pdf`
2. DAPO算法
3. GRPO算法
4. RULER benchmark

### 10.3 代码阅读顺序

```
1. quickstart.py → 理解Memory机制
2. recurrent/interface.py → 理解接口设计
3. recurrent/impls/memory.py → 理解核心实现
4. run_memory_7B_3090.sh → 理解训练配置
5. recurrent/generation_manager.py → 理解训练循环
6. verl/workers/reward_manager/thread.py → 理解Reward计算
```

---

**文档版本**: v1.0  
**适用资源**: 8×RTX 3090 (24GB)  
**预计时间**: 4周  
**最后更新**: 2025-06
