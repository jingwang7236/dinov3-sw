
# 📘 DINOv3 7B 分布式权重（distcp）合并的方法总结/避坑指南

## 一、distcp 是什么（一句话）
**PyTorch 分布式分片格式**，不是单个 `.pth`，而是一个目录：
- 包含：`.metadata` + 一堆 `.distcp` 分片
- 由 FSDP 训练自动生成
- **只能用官方 DCP API 读取，不能用 torch.load**

---

## 二、所有失败原因汇总（必背）
### ❌ 失败1：用 `torch.load()` 加载 distcp 目录
- 错误：`torch.load("xxx/distcp_dir")`
- 报错：`corrupted` / `invalid file` / `KeyError`
- 原因：`torch.load` 只认**单个文件**，不认目录
- ✅ 正确：用 `torch.distributed.checkpoint`

### ❌ 失败2：加载完整训练 checkpoint（OrderedDict）
- 错误：加载 `teacher_checkpoint.pth`（含完整训练状态）
- 报错：`AttributeError: 'OrderedDict' has no attribute 'is_leaf'`
- 原因：模型初始化只想要**纯参数 tensor**，不要整个字典
- ✅ 正确：只提取 `teacher`/`student` 下的权重

### ❌ 失败3：手写分片合并逻辑
- 错误：自己写 gather / 自己拼 state_dict
- 报错：权重错位、维度不匹配、结构错误
- 原因：distcp 分片规则复杂，**只能官方 API 解析**
- ✅ 正确：调用 `dcp.load + DefaultLoadPlanner + no_dist=True`

### ❌ 失败4：一次性全加载，爆内存/显存
- 错误：一次性加载整个 7B 权重
- 报错：OOM、内存溢出、进程被杀
- 原因：7B BF16≈14GB，4090 显存/内存扛不住
- ✅ 正确：用 **mmap 内存映射**，磁盘虚拟加载，内存占用<200MB

---

## 三、唯一成功方案（直接抄命令）
### 官方 DCP + mmap 低内存合并脚本（你的成功脚本）
```bash
python merge_distcp_to_pth.py \
  --src 你的distcp目录 \
  --dst 输出teacher_checkpoint.pth \
  --include-prefix model. \
  --strip-prefix model. \
  --wrap-key teacher \
  --mmap-dir /tmp
```

### 成功核心 4 点
1. **读 `.metadata`**：正确解析分片位置、大小、 dtype
2. **官方 API**：`dcp.load(no_dist=True, DefaultLoadPlanner)`，适配单机环境
3. **mmap 低内存**：磁盘映射替代内存，4090 稳跑
4. **只取模型权重**：过滤 optimizer/epoch 等无用项，输出 DINOv3 标准格式

### 输出格式（DINOv3 二阶段直接用）
```python
{"teacher": 模型权重state_dict}
```

---

## 四、以后遇到大模型权重，3条铁律
1. **distcp 目录：只用 `torch.distributed.checkpoint`，别用 `torch.load`**
2. **7B 级模型：必须用 `mmap` 低内存加载，否则必 OOM**
3. **训练 checkpoint：只提取 `teacher`/`student` 权重，别加载完整字典**
