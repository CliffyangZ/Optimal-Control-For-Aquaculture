# 鱼饲料检测模型优化分析 (YOLO v26)

## 执行概览

当前模型识别率不高的主要原因分析和优化方案。目标是通过系统性优化提升mAP50至≥0.75（Feed）和≥0.80（Fish）。

---

## 一、现状分析

### 1.1 当前框架配置

**训练框架**：
- 模型: YOLOv26s (小型模型，20M参数)
- 可用预训练: yolo26n.pt (5.3M nano), yolo26s.pt (20M small)
- 数据集: FishTracker_v2 (Roboflow)
- 训练方式: 5折交叉验证
- Epoch: 300
- Batch: 8
- Image Size: 640
- 数据增强: Mosaic (1.0), Mixup (0.1)
- 早停耐心: 50

**推理框架**：
- 置信度阈值: 0.70
- NMS: 默认参数
- 设备: CUDA
- 输出: JSON + 可视化视频

**质量门槛**：
- Feed: mAP50 ≥ 0.70, Recall ≥ 0.75
- Fish: mAP50 ≥ 0.75

### 1.2 潜在瓶颈

**模型结构层面**：
- ⚠️ YOLOv26s (20M参数) 相对较小，特征提取能力有限
- ⚠️ 水下环境复杂（光照变化、小目标）可能超出小模型能力
- ❌ 单一模型，没有集成策略
- ❌ 没有yolo26m/yolo26l的预训练权重，无法直接升级

**数据处理层面**：
- ❌ Batch=8 过小，梯度估计不稳定
- ❌ 数据增强可能不够针对水下拍摄的特殊场景
- ❌ 没有类别平衡策略
- ❌ 数据预处理可能不够优化

**推理层面**：
- ❌ 单阈值推理，没有动态调整
- ❌ 没有后处理优化 (NMS参数未调)
- ❌ 没有Test-Time Augmentation (TTA)
- ❌ 没有模型集成推理

**训练策略层面**：
- ❌ 没有预热策略
- ❌ 学习率策略可能不够激进
- ❌ 没有类别权重调整
- ❌ 没有Hard Example Mining

---

## 二、分阶段优化方案

### 第一阶段：快速收益（预期 +3-5% mAP）

#### 2.1.1 优化当前模型的初始化和训练

由于YOLOv26没有M/L预训练权重，采用以下策略：

**选项A**（推荐）：增强yolo26s的训练过程
```yaml
# Depolyment/src/config/training.yaml - 保持当前模型
model:
  weights: yolo26s.pt  # 继续用S，但优化训练
  batch: 16            # 关键：增加batch (见 2.1.2)
  imgsz: 832           # 关键：增加分辨率 (见 2.1.3)
  epochs: 400          # 给更多training iterations
```

**选项B**（可选试验）：从nano初始化，更多迭代训练
```yaml
model:
  weights: yolo26n.pt  # 从nano开始
  epochs: 500          # 更多epoch让nano学习更多特征
```

**优势**：
- 合理的参数大小和性能平衡
- 通过增加训练强度弥补模型大小的不足
- S版本推理速度保持合理

#### 2.1.2 增加Batch大小

```yaml
model:
  batch: 16  # 8 → 16
  # 如果显存不足，尝试 batch: 12
```

**优势**：
- 梯度估计更稳定
- 批量归一化效果更好
- 收敛更快

**前提**：需要有足够显存（RTX 3090/4090)

#### 2.1.3 提升图像分辨率

```yaml
model:
  imgsz: 832  # 640 → 832
  # 或 1024
```

**优势**：
- 更多图像细节被保留
- 对小目标（幼鱼/饲料粒）更敏感
- Feed和Fish都是相对小的目标，更高分辨率有帮助

**成本**：
- 内存×2.7倍 (640² → 832²)
- 推理速度×1.7倍

#### 2.1.4 调整NMS参数

在推理代码中添加NMS参数优化：

```python
# core/infer/detector.py - modify detect()
results = self._model(
    raw_frame, 
    conf=self._conf_threshold,
    iou=0.45,  # NMS IoU阈值，默认0.7，较低值删除重复检测
    verbose=False
)
```

**优势**：
- NMS IoU=0.45 对密集小目标更有效
- 减少重叠检测

### 第二阶段：数据增强优化（预期 +3-5% mAP）

#### 2.2.1 增强数据增强参数

```yaml
model:
  mosaic: 1.0      # 已启用
  close_mosaic: 10 # 保持
  mixup: 0.2       # 0.1 → 0.2 (增加混合强度)
  # 添加以下参数到 model.train() 调用
  hsv_h: 0.015     # 色调变化
  hsv_s: 0.7       # 饱和度变化
  hsv_v: 0.4       # 亮度变化
  degrees: 10      # 旋转角度
  translate: 0.1   # 平移
  scale: 0.5       # 缩放
  flipud: 0.5      # 上下翻转
  fliplr: 0.5      # 左右翻转
```

**针对水下环境**：
- 需要考虑光照变化
- 水的折射和扭曲效应
- 不同角度的鱼饲料外观

#### 2.2.2 自定义数据预处理

创建新文件 `core/train/augmentation.py`：

```python
class AquacultureAugmentation:
    """针对水下环境的增强策略"""
    
    @staticmethod
    def apply_underwater_effects(image):
        """模拟水下效果"""
        # 1. 加入色温变化（蓝绿色偏移）
        # 2. 加入光照不均匀
        # 3. 加入轻微扭曲（水波纹）
        pass
    
    @staticmethod
    def equalize_histogram(image):
        """直方图均衡化改善对比度"""
        pass
```

### 第三阶段：高级推理策略（预期 +2-4% mAP）

#### 2.3.1 Test-Time Augmentation (TTA)

```python
# 在推理时使用多张变形的图像
def detect_with_tta(self, frame):
    """使用TTA增强推理"""
    augmented_frames = [
        frame,
        cv2.flip(frame, 1),  # 水平翻转
        cv2.flip(frame, 0),  # 垂直翻转
        cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE),
    ]
    
    all_detections = []
    for aug_frame in augmented_frames:
        det = self._model(aug_frame)
        all_detections.append(det)
    
    # 合并结果，进行NMS
    return self._merge_detections(all_detections)
```

**优势**：
- 对旋转和翻转更鲁棒
- 提升recall和mAP
- 缺点：推理速度×4

#### 2.3.2 模型集成（Ensemble）

```python
class EnsembleDetector:
    """多模型集成推理"""
    
    def __init__(self, model_paths: list[str], weights: list[float] = None):
        self.models = [YOLO(p) for p in model_paths]
        self.weights = weights or [1/len(model_paths)] * len(model_paths)
    
    def detect(self, frame):
        all_results = []
        for model in self.models:
            results = model(frame)
            all_results.append(results)
        
        # 加权合并
        return self._merge_with_weights(all_results, self.weights)
```

**策略**：
- 不同大小的模型 (S, M, L)
- 不同分辨率训练的模型
- 最后一层特征层融合

### 第四阶段：训练策略优化（预期 +2-3% mAP）

#### 2.4.1 类别平衡和权重

```python
# core/train/trainer.py - 在 model.train() 中添加
results = model.train(
    # ...
    cls_weight=[1.0, 1.2, 0.8, 0.9],  # 对应: fish, feed, debris, lumpfish
    # feed可以给更高权重（1.2）确保识别率
)
```

#### 2.4.2 Hard Example Mining

```python
# 在训练循环中识别困难样本
def identify_hard_examples(val_results):
    """识别低置信度的检测"""
    hard_samples = [
        (img_path, result) 
        for img_path, result in val_results 
        if result.max_conf < 0.5 or result.has_missed_detections()
    ]
    # 在下一个epoch重新采样这些样本
    return hard_samples
```

#### 2.4.3 学习率调度

```yaml
model:
  lr0: 0.001     # 初始学习率（从0.01降低）
  lrf: 0.0001    # 最终学习率比例
  # 使用余弦退火，中间有热重启
```

### 第五阶段：YOLO v26 特定优化（预期 +2-4% mAP）

由于YOLOv26没有M/L预训练权重，采用以下高级策略来弥补模型大小的限制：

#### 2.5.1 知识蒸馏（Knowledge Distillation）

从一个更大的模型（如YOLOv8m）蒸馏知识到yolo26s：

```python
# 伪代码 - core/train/distillation.py
class KnowledgeDistillation:
    def __init__(self, teacher_model_path, student_model_path):
        self.teacher = YOLO(teacher_model_path)  # 大型教师模型
        self.student = YOLO(student_model_path)  # yolo26s学生模型
        self.temperature = 4.0
        self.alpha = 0.7  # 蒸馏损失权重
    
    def train_with_distillation(self, train_loader):
        # 组合KL divergence损失和标准检测损失
        # 让学生模型学习教师模型的概率分布
        pass
```

**优势**：
- 让小模型学习大模型的特征
- 通常可以提升 +2-3% mAP
- 推理速度和内存不变

#### 2.5.2 多分支架构扩展

```python
# 在yolo26s基础上添加轻量级注意力机制
class EnhancedYOLOv26s:
    """添加SE或CA注意力到yolo26s"""
    def __init__(self):
        self.base_model = YOLO('yolo26s.pt')
        # 在backbone中插入注意力层
        # 成本：+10%参数，可获得 +1-2% mAP
```

#### 2.5.3 级联检测（Cascade Detection）

```python
# 两阶段检测：粗定位 → 精细检测
class CascadeYOLODetector:
    """使用两个yolo26s级联提高准确率"""
    def __init__(self):
        self.stage1 = YOLO('yolo26s.pt')  # 快速粗定位
        self.stage2 = YOLO('yolo26s_refine.pt')  # 精细定位
    
    def detect(self, frame):
        # 阶段1：快速检测和ROI提取
        rois = self.stage1(frame)
        
        # 阶段2：在ROI区域内精细检测
        refined = self.stage2(rois)
        
        return refined
```

**优势**：
- 推理时间加倍，但准确率 +2-3%
- 适合离线或高准确率场景

---

## 三、执行优先级和预期收益

| 优先级 | 优化方案 | 实现难度 | 预期收益 | 推荐 |
|-------|---------|---------|---------|------|
| 1️⃣ | Batch 增加到16 | 简单 | +1.5% | ✅ 立即执行 |
| 2️⃣ | 图像分辨率832 | 简单 | +2% | ✅ 立即执行 |
| 3️⃣ | 增加Epoch到400 | 简单 | +1.5% | ✅ 立即执行 |
| 4️⃣ | NMS参数调整 | 简单 | +0.5% | ✅ 立即执行 |
| 5️⃣ | 增强数据增强 | 中等 | +3% | ✅ 优先级高 |
| 6️⃣ | 类别权重调整 | 简单 | +1% | ✅ 优先级高 |
| 7️⃣ | 自定义模型架构(yolo26m) | 困难 | +3-4% | ⏳ 后期优化 |
| 8️⃣ | TTA推理 | 中等 | +3% | ⏳ 后期优化 |
| 9️⃣ | 模型集成 (多个yolo26) | 困难 | +4% | ⏳ 后期优化 |

**预期总收益**：+5-8% mAP（分阶段执行）

---

## 四、实施路线图

### Phase 1: 快速基线改进（1-2周）
```
Week 1:
  □ 升级到YOLOv8m + 增加Batch/Resolution
  □ 调整NMS参数
  □ 重新训练（5折）
  □ 评估新性能指标
```

### Phase 2: 数据和训练优化（2-3周）
```
Week 2:
  □ 实现水下环境数据增强
  □ 调整类别权重
  □ 调整超参数 (学习率, Mixup等)
  □ 重新训练并记录WandB指标
```

### Phase 3: 推理优化（1-2周）
```
Week 3:
  □ 实现TTA推理
  □ 评估性能-速度权衡
  □ 优化NMS和置信度阈值
```

### Phase 4: 高级优化（2-4周）
```
Week 4:
  □ 训练多个模型变体
  □ 实现Ensemble推理
  □ 最终性能基准测试
```

---

## 五、监控指标

需要在WandB中追踪这些指标：

### 性能指标
- mAP@0.5 (整体和按类)
- mAP@0.75
- Recall (按类)
- Precision (按类)
- F1-Score

### 诊断指标
- 假正样本 (False Positives)
- 漏检样本 (False Negatives)
- 置信度分布
- 检测框大小分布

### 推理指标
- 推理延时 (ms/frame)
- 吞吐量 (fps)
- 显存占用

---

## 六、快速启动指南

### 立即可执行的更改

**1. 更新Depolyment/src/config/training.yaml**：
```yaml
model:
  weights: yolo26s.pt  # 保持S版本，通过训练优化
  variant: yolo26s
  batch: 16            # 从 8 升级 (关键优化)
  imgsz: 832           # 从 640 升级 (关键优化)
  epochs: 400          # 从 300 升级 (更多迭代)
  mixup: 0.2           # 从 0.1 升级
```

**2. 更新core/infer/detector.py**：
```python
# Line 43
results = self._model(
    raw_frame,
    conf=self._conf_threshold,
    iou=0.45,  # 添加NMS参数
    verbose=False
)
```

**3. 在Depolyment/src/config/training.yaml的model.train()中添加**：
```yaml
model:
  # ... 现有配置
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10
  translate: 0.1
  scale: 0.5
  flipud: 0.5
  fliplr: 0.5
```

**4. 启动新训练**：
```bash
python train.py download  # 如果需要更新数据集
python train.py train     # 使用新配置训练
python train.py evaluate  # 评估5折结果
```

---

## 七、故障排查

**问题1**：增加batch/分辨率后显存不足
```
解决方案：
  - 降低batch: 16 → 12 或 8
  - 降低分辨率: 832 → 768 或 704
  - 使用model.train(..., device=0) 单GPU
  - 启用混合精度训练: amp=True (自动混合精度)
```

**问题2**：训练速度太慢
```
解决方案：
  - 降低 mixup: 0.2 → 0.1
  - 关闭 mosaic: 1.0 → 0.7
  - 早期停止 patience: 50 → 30
```

**问题3**：性能没有提升
```
解决方案：
  - 检查数据增强是否过度（dropout）
  - 验证数据集是否有质量问题
  - 尝试减小学习率
  - 检查类别不平衡问题
```

---

## 八、参考资源

- YOLOv8 官方文档: https://docs.ultralytics.com/
- 数据增强最佳实践: https://github.com/albumentations-team/albumentations
- 水下目标检测相关论文
- 模型集成策略文献

---

## 九、YOLOv26 优化总结

### 推荐执行顺序（YOLO v26特定）

```
第1周：基础优化（预期 +3-5%）
  ├─ 增加 batch: 8 → 16
  ├─ 增加 imgsz: 640 → 832
  ├─ 增加 epochs: 300 → 400
  └─ 调整NMS参数: iou=0.45

第2周：数据和训练优化（预期 +3%）
  ├─ 增强数据增强参数
  ├─ 调整类别权重 (feed: 1.2)
  ├─ 调整学习率: lr0=0.001
  └─ 重新训练5折

第3周：推理优化（预期 +2%）
  ├─ 实现TTA推理
  ├─ 优化置信度阈值
  └─ 性能-速度权衡评估

第4周：高级优化（可选，预期 +2-3%）
  ├─ 知识蒸馏（从v8m）
  └─ 模型集成（多个v26）
```

### 关键指标监控

```yaml
# WandB中必须监控这些指标
Key_Metrics:
  - mAP@0.5 (feed): 目标 ≥ 0.75
  - mAP@0.5 (fish): 目标 ≥ 0.80
  - Recall (feed): 目标 ≥ 0.75
  - Precision (feed): 维持 > 0.70
  - Inference_time: 监控推理延时
  - GPU_Memory: 监控显存占用
```

### 注意事项

⚠️ **YOLOv26 vs YOLOv8**：
- v26是相对较新的版本，可能文档较少
- 如果遇到问题，查看Ultralytics官方文档的YOLO部分
- 某些YOLOv8的优化可能不完全适用于v26

✅ **立即行动**：
1. 备份当前配置
2. 更新Depolyment/src/config/training.yaml（batch, imgsz, epochs）
3. 运行第一个fold测试验证显存够用
4. 开始完整的5折训练

---

**预计通过以上优化，可将mAP提升 5-8%，达到目标阈值。**
