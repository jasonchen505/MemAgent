# MemAgent 增量学习笔记

> 基于前两轮学习（INTERVIEW_PREP.md、INTERVIEW_DEEP_DIVE.md）
> 记录复现过程中新学到的点、深入理解、实践感悟

---

## 一、从理论到实践的增量理解

### 1.1 显存管理的实践理解

**前两轮认知**：
- 知道需要开启param_offload、optimizer_offload
- 知道gradient_checkpointing可以减少显存

**实践增量**：
```bash
# 实际配置中发现的关键点
actor_rollout_ref.actor.fsdp_config.param_offload=True        # 参数offload到CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True    # 优化器状态offload到CPU
actor_rollout_ref.rollout.gpu_memory_utilization=0.4          # vLLM只用40%显存
```

**深入理解**：
1. **param_offload的代价**：每次forward/backward都要从CPU加载参数，增加约2-3x延迟
2. **gpu_memory_utilization的平衡**：
   - 太高(0.6+)：训练时可能OOM
   - 太低(0.3-)：vLLM batch size太小，生成效率低
   - 3090建议：0.4-0.5
3. **FSDP的作用**：
   - 将参数、梯度、优化器状态分片到多卡
   - fsdp_size=8表示8卡全参与分片
   - 每张卡只存储1/8的完整状态

### 1.2 vLLM与训练框架的交互

**前两轮认知**：
- vLLM用于高效推理
- 训练时需要生成response

**实践增量**：

```python
# vLLM在训练中的角色（generation_manager.py:146-148）
with _timer('mt_gen', timing_raw):
    gen_output = self.generate_with_graceful_padding(
        input_ids, attention_masks, position_ids, meta_info_gen
    )
```

**关键发现**：
1. **vLLM不是一直运行的**：
   - 训练时：vLLM用于生成response（rollout阶段）
   - 训练更新时：vLLM休眠，释放显存给训练
   - 这就是`wake_up()`和`sleep()`的作用

2. **显存竞争**：
   - vLLM需要显存做KV Cache
   - 训练需要显存做梯度计算
   - 两者需要平衡，所以gpu_memory_utilization不能太高

3. **graceful_padding的工程细节**：
   ```python
   # 问题：batch_size不能被world_size整除
   # 解决：用"Hello."作为padding
   def graceful_padding(bsz, group_nums):
       # Example: bsz=7, group_nums=3
       # no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
       # padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
   ```

### 1.3 训练循环的深入理解

**前两轮认知**：
- 理解RAgent的action/update/done循环
- 理解Memory的逐步累积

**实践增量**：

```python
# generation_manager.py:120-159
def run_llm_loop(self, gen_batch, timing_raw):
    active_num_list = []  # 追踪每轮active样本数
    gen_output_list = []  # 存储每轮的I/O
    
    self.agent.start(gen_batch, timing_raw)
    while not self.agent.done():
        # 1. 准备输入
        messages, meta_info_gen = self.agent.action()
        
        # 2. 生成
        gen_output = self.generate_with_graceful_padding(...)
        
        # 3. 更新状态
        gen_output = self.agent.update(gen_output)
        gen_output_list.append(gen_output)
    
    # 4. 收集最终结果
    final_mask, sample_index = self.agent.end()
    return DataProto.concat(gen_output_list), final_mask, sample_index
```

**关键发现**：
1. **active_num_list的作用**：
   - 追踪每轮有多少样本还在处理
   - 随着chunk消耗完，active样本逐渐减少
   - 用于调试和监控

2. **final_mask的重要性**：
   - 只有final_mask=True的样本会被reward
   - 中间轮的生成不会被直接优化
   - 但中间轮的质量影响最终答案

3. **DataProto.concat的工程意义**：
   - 将多轮生成的结果concatenate
   - 保持sample_index的映射关系
   - 用于后续的policy optimization

---

## 二、参数调优的实践感悟

### 2.1 rollout.n的选择

**前两轮认知**：
- GRPO需要n个response做组内比较
- 原配置n=16

**实践增量**：
```bash
# 原配置
actor_rollout_ref.rollout.n=16  # 每prompt生成16个response

# 3090配置
actor_rollout_ref.rollout.n=4   # 只能生成4个
```

**深入理解**：
1. **n与训练稳定性的关系**：
   - n越大，组内比较越稳定，方差越小
   - n太小(如2)，reward信号噪声大，训练不稳定
   - 3090上n=4是底线，建议n=8如果显存允许

2. **n与显存的关系**：
   - n=16意味着每个prompt要生成16个response
   - vLLM的KV Cache需要16x显存
   - 这是显存的主要消耗之一

3. **实际调优建议**：
   ```bash
   # 如果OOM，先减n
   actor_rollout_ref.rollout.n=2  # 最小值
   # 如果训练不稳定，尝试增加n（需要减小其他参数）
   ```

### 2.2 train_batch_size的选择

**前两轮认知**：
- 原配置train_batch_size=128
- 3090需要减小

**实践增量**：

**深入理解**：
1. **batch_size与梯度估计的关系**：
   - 大batch：梯度估计稳定，但更新慢
   - 小batch：更新快，但梯度噪声大
   - RL训练对batch_size更敏感（因为reward信号稀疏）

2. **batch_size与rollout.n的配合**：
   ```python
   # 实际生成的response数 = train_batch_size × rollout.n
   # 原配置：128 × 16 = 2048个response
   # 3090配置：16 × 4 = 64个response
   ```

3. **3090的建议**：
   - 最小可行：train_batch_size=4, rollout.n=2
   - 推荐配置：train_batch_size=16, rollout.n=4
   - 显存允许：train_batch_size=32, rollout.n=8

### 2.3 ppo_max_token_len_per_gpu的选择

**前两轮认知**：
- 控制每GPU处理的最大token数
- 原配置16384

**实践增量**：
```bash
# 原配置
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384

# 3090配置
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
```

**深入理解**：
1. **这个参数的作用**：
   - 控制PPO更新时每GPU的token数
   - 影响梯度累积的粒度
   - 太大可能OOM，太小训练效率低

2. **与use_dynamic_bsz的配合**：
   ```bash
   actor_rollout_ref.actor.use_dynamic_bsz=True
   ```
   - 动态batch size：根据token数自动调整batch
   - 实际batch_size可能比设置的更小
   - 更稳定的训练，但可能更慢

---

## 三、工程实践的增量理解

### 3.1 Ray集群的管理

**前两轮认知**：
- Ray用于分布式计算
- ActorPool用于分布式Reward计算

**实践增量**：

```bash
# 启动单节点Ray
ray start --head

# 查看Ray状态
ray status

# 停止Ray
ray stop

# 多节点集群（原配置需要）
# 节点1：ray start --head --port=6379
# 节点2：ray start --address='node1:6379'
```

**关键发现**：
1. **Ray的资源管理**：
   - 自动检测GPU资源
   - Actor会调度到有资源的节点
   - RewardActorPool的NodeAffinitySchedulingStrategy确保Actor在GPU节点

2. **Ray的常见问题**：
   ```bash
   # 问题：Ray启动失败
   # 解决：清理旧状态
   ray stop
   rm -rf /tmp/ray
   ray start --head
   
   # 问题：Actor创建失败
   # 解决：检查资源
   ray status  # 查看可用资源
   ```

### 3.2 Checkpoint的保存与恢复

**前两轮认知**：
- 知道有checkpoint机制
- 知道save_freq控制保存频率

**实践增量**：

```bash
# 保存配置
trainer.save_freq=5                    # 每5个epoch保存
trainer.default_local_dir=$PROJ_DIR    # 保存路径

# 恢复训练
python3 -m verl.trainer.main_ppo \
    ... \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$PROJ_DIR
```

**关键发现**：
1. **Checkpoint的内容**：
   ```
   checkpoints/memory_agent/7B_3090/
   ├── global_step_100/
   │   ├── actor/
   │   ├── rollout/
   │   └── metadata
   ├── global_step_200/
   └── latest
   ```

2. **恢复训练的注意事项**：
   - 必须使用相同的配置（除了resume相关参数）
   - 数据集必须存在且路径正确
   - 恢复时会自动加载latest checkpoint

3. **手动指定checkpoint**：
   ```bash
   # 从特定checkpoint恢复
   trainer.resume_mode=resume_path
   trainer.resume_from_path=/path/to/global_step_100
   ```

### 3.3 日志与监控

**前两轮认知**：
- 知道用wandb记录
- 知道有timer机制

**实践增量**：

```python
# 关键日志点（generation_manager.py:158）
logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")

# memory.py:265-288的log_step
def log_step(self, gen_output):
    logger.info(f"[MESSAGE] {decoded_message}")
    logger.info(f"[RESPONSE] {decoded_response}")
```

**关键发现**：
1. **ACTIVE_TRAJ_NUM的意义**：
   - 每轮active样本数
   - 应该逐渐减少（样本完成处理）
   - 如果不减少，说明done()逻辑有问题

2. **Message/Response的日志**：
   - 只打印第一个样本（idx=0）
   - 用于快速检查生成质量
   - 可以观察Memory是否在正确累积

3. **Wandb的关键指标**：
   ```python
   # 训练时关注
   - reward/mean: 平均reward，应该上升
   - kl/mean: KL散度，应该稳定
   - response_length_mean: 生成长度，应该稳定
   - active_traj_num: 活跃样本数，应该逐渐减少
   ```

---

## 四、问题排查的实践增量

### 4.1 OOM问题的排查

**前两轮认知**：
- 知道需要减小batch_size、开启offload

**实践增量**：

**排查步骤**：
```bash
# 1. 确认OOM发生在哪个阶段
# 训练阶段：减ppo_max_token_len_per_gpu
# 生成阶段：减gpu_memory_utilization

# 2. 查看显存使用
nvidia-smi -l 1

# 3. 逐步减小参数
# 先减到最小可行配置
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
data.train_batch_size=4
actor_rollout_ref.rollout.n=2
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096

# 确认能跑通后，逐步增加
```

**关键发现**：
1. **OOM的常见原因**：
   - vLLM显存不足：减gpu_memory_utilization
   - 训练显存不足：减ppo_max_token_len_per_gpu
   - 梯度累积过多：减train_batch_size

2. **显存碎片化**：
   - 开启`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 可以减少碎片化导致的OOM

### 4.2 训练不收敛的排查

**前两轮认知**：
- 知道检查数据、Reward、梯度

**实践增量**：

**排查步骤**：
```python
# 1. 检查Reward是否正常
# 在ThreadRewardManager中（thread.py:248）
print(sequences_strs[0])  # 打印第一个样本
print(reward_tensor[0].sum())  # 打印reward

# 2. 检查生成质量
# 在MemoryAgent.log_step中（memory.py:265）
logger.info(f"[MESSAGE] {decoded_message}")
logger.info(f"[RESPONSE] {decoded_response}")

# 3. 检查active样本数
# 在generation_manager.py:158
logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
```

**关键发现**：
1. **Reward全为0的常见原因**：
   - 模型没有生成`\boxed{answer}`格式
   - 验证器解析失败
   - 解决：检查生成内容，调整prompt

2. **Memory质量差的原因**：
   - chunk_size太大，信息压缩过度
   - 训练不足，还没学会压缩
   - 解决：减小chunk_size，增加训练

### 4.3 性能突然下降的排查

**前两轮认知**：
- 知道检查模型加载、tokenizer

**实践增量**：

**排查清单**：
```python
# 1. 检查checkpoint是否损坏
# 尝试从之前的checkpoint恢复

# 2. 检查数据是否变化
# 确认数据路径没变

# 3. 检查配置是否变化
# 对比当前配置和之前的配置
```

**关键发现**：
1. **常见的"假性下降"**：
   - 验证集变化：确保验证集固定
   - 评估代码变化：确保评估逻辑一致
   - 随机性：多次评估取平均

2. **真实的性能下降**：
   - 过拟合：训练太久，泛化能力下降
   - 灾难性遗忘：学习新能力时忘记旧能力
   - 解决：early stopping，减少学习率

---

## 五、代码理解的增量

### 5.1 TokenTemplate的深入理解

**前两轮认知**：
- 知道TokenTemplate避免重复编解码
- 知道性能提升20x

**实践增量**：

```python
# utils.py:25-113
class TokenTemplate:
    def __init__(self, template, tokenizer):
        # 1. 解析模板中的{keyword}
        # 2. 将非keyword部分tokenize
        # 3. 存储为token sections
    
    def format(self, **kwargs):
        # 直接在token空间拼接
        # 避免：decode → format → encode
        formatted_parts = []
        for i, k in enumerate(self.keywords):
            formatted_parts.append(self.token_sections[i])
            formatted_parts.append(kwargs[k])
        formatted_parts.append(self.last_section)
        return torch.cat(formatted_parts)
```

**深入理解**：
1. **为什么性能提升这么多**：
   - decode需要遍历token，O(n)时间
   - encode需要tokenizer推理，更慢
   - 直接拼接token，O(1)时间

2. **限制性**：
   - 只能处理token级别的拼接
   - 不支持动态文本处理（如条件判断）
   - 需要预先知道所有keyword

### 5.2 AsyncRAgent的深入理解

**前两轮认知**：
- 知道Async模式适合API调用
- 知道"Agent as a Function"的理念

**实践增量**：

```python
# async_memory.py:96-175
async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
    memory = None
    while True:
        if step * self.config.chunk_size >= context_length:
            break
        
        # 直接调用API，无需处理tokenization
        completions = await self.proxy.get_chat_completions(
            messages=conversation,
            **kwargs
        )
        memory = completions.choices[0].message.content
        step += 1
    
    return AsyncOutput(conversations, sample_index, final_mask, timing_raw)
```

**深入理解**：
1. **与Sync模式的区别**：
   - Sync：批量处理，需要padding，复杂
   - Async：单样本处理，直接API调用，简单

2. **Async的优势**：
   - 代码简洁，无需处理batch
   - 天然支持并发
   - 适合推理服务

3. **Async的限制**：
   - 不适合训练（训练需要batch优化）
   - 每次API调用有网络延迟
   - 需要async/await支持

### 5.3 RewardActorPool的深入理解

**前两轮认知**：
- 知道用Ray Actor做分布式Reward计算
- 知道Least Requests负载均衡

**实践增量**：

```python
# thread.py:84-159
class RewardActorPool:
    def __init__(self):
        # 在每个GPU节点创建Actor
        for node in ray.nodes():
            actor = RewardActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,  # 硬亲和性
                )
            ).remote()
        
        # Least Requests负载均衡
        self.weighted_actors = [[0, i, actor] for i, actor in enumerate(actors)]
        heapq.heapify(self.weighted_actors)
```

**深入理解**：
1. **NodeAffinitySchedulingStrategy的作用**：
   - 确保Actor在指定节点
   - 避免跨节点数据传输
   - 提高计算效率

2. **Least Requests的优势**：
   - 动态平衡负载
   - 避免某个Actor过载
   - heap实现，O(log n)复杂度

3. **batch_reward的设计**：
   ```python
   def batch_reward(self, method, args, **const):
       # 批量处理，减少Ray调用开销
       futures = []
       for i, kwargs in args:
           fut = self.executor.submit(method, **kwargs)
           futures.append((i, fut))
       # 等待所有完成
       for i in range(len(futures)):
           idx, fut = futures[i]
           futures[i] = (idx, fut.result())
       return futures
   ```

---

## 六、实验设计的增量理解

### 6.1 消融实验的设计

**前两轮认知**：
- 知道需要消融chunk_size、rollout.n等
- 知道需要对比无RL训练

**实践增量**：

**消融实验优先级**：
```python
# P0: 必做
1. 无RL训练的baseline（只用prompt engineering）
2. 不同chunk_size: [2000, 5000, 10000]
3. 不同rollout.n: [2, 4, 8]

# P1: 尽量做
4. 不同train_batch_size: [4, 8, 16]
5. 不同learning_rate: [1e-7, 1e-6, 1e-5]

# P2: 资源允许
6. 不同max_memorization_length
7. 不同temperature
```

### 6.2 评估指标的选择

**前两轮认知**：
- 知道用RULER benchmark
- 知道训练/测试验证器差异

**实践增量**：

**关键指标**：
```python
# 1. 基础指标
- RULER_HQA准确率：最直接的性能指标
- 训练reward：训练过程的信号

# 2. 效率指标
- 训练时间：复现成本
- 显存使用：资源需求

# 3. 泛化指标
- 不同长度的性能：外推能力
- OOD任务性能：泛化能力
```

### 6.3 结果对比的方法

**前两轮认知**：
- 知道要与原论文对比
- 知道要量化差异

**实践增量**：

**对比维度**：
```python
# 1. 绝对性能
your_score vs original_score
# 差异<5%：优秀
# 差异5-10%：良好
# 差异>10%：需要分析原因

# 2. 训练曲线
# 对比reward上升速度
# 对比收敛稳定性

# 3. 泛化能力
# 对比不同长度的性能衰减
# 对比OOD任务性能
```

---

## 七、业务理解的增量

### 7.1 成本效益分析

**前两轮认知**：
- 知道训练成本高
- 知道推理有延迟

**实践增量**：

**实际成本估算**：
```python
# 训练成本（3090）
# 8卡3090，电费约0.5元/度，总功耗约3KW
# 每天电费：3 × 24 × 0.5 = 36元
# 训练25天：36 × 25 = 900元

# 对比A100
# 32卡A100，总功耗约15KW
# 每天电费：15 × 24 × 0.5 = 180元
# 训练3天：180 × 3 = 540元

# 结论：3090训练成本更高（时间更长）
# 但硬件成本低（3090 vs A100）
```

### 7.2 实际应用场景

**前两轮认知**：
- 知道适合长文档QA
- 知道不适合实时对话

**实践增量**：

**应用场景细化**：
```python
# 1. 法律文档分析
# 优势：长文本，信息密集，需要全局理解
# 挑战：专业术语，格式复杂
# 成本：每文档约10-20元（API调用）

# 2. 学术论文综述
# 优势：多篇论文，需要综合信息
# 挑战：引用关系，知识整合
# 成本：每篇约5-10元

# 3. 代码库理解
# 优势：代码长，需要跨文件理解
# 挑战：代码结构，依赖关系
# 成本：中等
```

### 7.3 优化方向

**前两轮认知**：
- 知道可以减小chunk_size
- 知道可以并行化

**实践增量**：

**优化优先级**：
```python
# P0: 立即可做
1. 减小chunk_size：减少单次调用延迟
2. 限制最大长度：减少总调用次数

# P1: 需要工程投入
3. 并行化chunk处理：需要修改代码
4. 缓存机制：增加存储，减少重复计算

# P2: 需要算法改进
5. 自适应chunk_size：根据内容动态调整
6. 更好的压缩算法：提升Memory质量
```

---

## 八、自测与验证

### 8.1 理解验证

**完成以下任务验证理解**：

- [ ] 能解释param_offload的代价和收益
- [ ] 能说明vLLM在训练中的角色
- [ ] 能列出至少3个OOM的解决方法
- [ ] 能解释rollout.n与训练稳定性的关系
- [ ] 能说明checkpoint的保存和恢复流程

### 8.2 实践验证

**完成以下任务验证实践能力**：

- [ ] 能成功运行quickstart.py
- [ ] 能成功运行小规模训练（1-2 epochs）
- [ ] 能监控训练状态（显存、reward、active样本数）
- [ ] 能从checkpoint恢复训练
- [ ] 能独立排查OOM问题

### 8.3 代码验证

**完成以下任务验证代码理解**：

- [ ] 能解释TokenTemplate的工作原理
- [ ] 能解释AsyncRAgent与RAgent的区别
- [ ] 能解释RewardActorPool的负载均衡策略
- [ ] 能修改chunk_size并重新训练
- [ ] 能添加新的日志输出

---

## 九、后续学习计划

### 9.1 短期目标（1-2周）

1. 完成quickstart验证
2. 成功运行小规模训练
3. 理解所有训练参数

### 9.2 中期目标（3-4周）

1. 完成10 epochs训练
2. 评估训练效果
3. 与原论文结果对比

### 9.3 长期目标（1-2月）

1. 理解所有核心代码
2. 能独立修改和优化
3. 能扩展到新任务

---

## 十、参考资料

### 10.1 官方文档

- verl: https://github.com/volcengine/verl
- vLLM: https://docs.vllm.ai/
- Ray: https://docs.ray.io/

### 10.2 论文

- MemAgent: paper/paper.pdf
- DAPO: https://arxiv.org/abs/2403.14520
- GRPO: https://arxiv.org/abs/2402.03300

### 10.3 代码

- 训练脚本: run_memory_7B_3090.sh
- 核心实现: recurrent/impls/memory.py
- 接口定义: recurrent/interface.py

---

**文档版本**: v1.0  
**基础**: INTERVIEW_PREP.md, INTERVIEW_DEEP_DIVE.md  
**更新**: 持续更新中  
**最后更新**: 2025-06
