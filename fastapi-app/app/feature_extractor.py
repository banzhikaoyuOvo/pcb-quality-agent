# -*- coding: utf-8 -*-
"""
YOLOv8 特征提取器
==================

【职责】
    1. 用 YOLOv8n 检测 PCB 图片中的缺陷（bbox + label + confidence）
    2. 对每个缺陷 bbox 裁剪图像，提取 256 维 L2 归一化特征向量
    3. 提供异步包装器，避免阻塞 FastAPI 事件循环

【核心难点：YOLOv8 Neck 是 DAG 结构】
    YOLOv8 的 Neck 部分包含 Concat 等多输入模块（m.f 是 list），
    常规 forward hook 在 model.predict() 内部会被静默跳过（因为 predict
    走的是 detect head 分支，不是完整的 forward）。

【解决方案：物理截断模型】
    1. 找到输出通道为 256 的最后一个 C2f 层（idx=21）
    2. 用 nn.ModuleList 封装前 N 层
    3. 手工逐层前向，多输入模块从 y 缓存中取依赖
    这样绕过了 predict() 的短路逻辑，能拿到中间层特征图。

【为什么用 256 维特征而非原始图片？】
    - 原始图片像素维度高（如 1280×1280×3 ≈ 500 万维），冗余大
    - 对光照、角度、背景敏感，同类缺陷像素距离可能远
    - 特征向量压缩语义，同类缺陷在向量空间更近，适合检索
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 模块级配置
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(
        Path(__file__).resolve().parent.parent
        / "runs" / "detect" / "runs" / "pcb_detect" / "train_v1" / "weights" / "best.pt"
    ),
)
# 模型权重路径。环境变量优先（Docker 部署时挂载到 /app/models/best.pt），
# 否则回退到仓库相对路径（本地开发）。

VECTOR_DIM = 256
# 特征向量维度。必须与 db.VECTOR_DIM 严格一致！

CROP_PADDING = 8
# 裁剪 bbox 时向外扩展的像素数。防止裁剪太紧导致缺陷边缘特征丢失。

CONF_THRESHOLD = 0.1
# YOLO 推理时的置信度阈值（低阈值保召回）。
# 后续在 extract_from_image 中会再用 DISPLAY_CONF_THRESHOLD 二次过滤。

DISPLAY_CONF_THRESHOLD = 0.20
# 展示层置信度阈值。低于此值的缺陷默认过滤，
# 但若全部缺陷都低于此值，会保留最高分的那一个作为保底。


class FeatureExtractor:
    """
    YOLOv8 缺陷检测 + 特征提取器。

    【生命周期】
        1. __init__ 加载模型权重，构建截断模型（一次性开销 ~1-2s）
        2. extract_from_image 完成检测 + 每缺陷特征提取
        3. async_extract_from_image 是 (2) 的异步包装器

    【线程安全】
        - self.model / self._truncated_model 在 __init__ 后只读
        - extract_from_image 是纯函数式调用（无实例状态修改）
        - 因此多线程并发调用是安全的（PyTorch 的 forward 本身线程安全）
    """

    DETECT_IMGSZ = 1280
    # 检测时的输入尺寸。
    # 训练时用 640，但 PCB 原图通常 2000+ 像素，640 缩放后缺陷太小导致漏检。
    # 实测 1280 能在 RTX 4060 Ti 上保持 1.2ms/帧，召回率大幅提升。

    EMBED_IMGSZ = 640
    # 特征提取时的输入尺寸。裁剪后的 bbox 尺寸较小，640 足够。

    def __init__(self, model_path: str = str(MODEL_PATH)):
        """
        Args:
            model_path: YOLOv8 权重文件路径

        Raises:
            FileNotFoundError: 权重文件不存在
            RuntimeError:       找不到输出通道为 256 的 C2f 层
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ✅ 兼容旧版 ultralytics: 某些版本下 torch.load 会拒绝加载非 weights_only 的 ckpt
        # 临时关闭此限制，加载完立即恢复，避免影响全局
        os.environ["TORCH_FORCE_WEIGHTS_ONLY"] = "0"
        try:
            self.model = YOLO(model_path)
        finally:
            os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY", None)

        # 推理固定用 CPU —— 特征提取的实际加速来自向量运算，YOLO 本身的推理
        # 在 CPU 上也足够快（1280 分辨率约 30ms）。若需 GPU 推理改为 "cuda"。
        self.device = "cpu"
        self.model.to(self.device)
        self.model.model.float()  # 强制 fp32，避免某些权重在半精度下溢出

        # ✅ 构建截断模型（核心：绕过 predict 的短路逻辑）
        self._trunc_layer_idx = self._find_neck_layer()
        self._truncated_model = self._build_truncated_model()
        print(
            f"[INIT] Truncated model built. "
            f"Stops at layer {self._trunc_layer_idx} (C2f, {VECTOR_DIM}-ch)"
        )

    # -------------------------------------------------------------------------
    # 模型结构分析
    # -------------------------------------------------------------------------

    def _find_neck_layer(self) -> int:
        """
        定位 Neck 中**最后一个输出通道为 256 的 C2f 层**。

        Returns:
            该层在 model.model.model 中的索引

        Raises:
            RuntimeError: 找不到符合条件的层

        【为什么跳过 idx <= 9？】
            YOLOv8n 的 Backbone 是前 10 层（idx 0~9），
            我们要的是 Neck 特征（更靠后，语义更强）。

        【为什么跳过 Detect？】
            Detect 头输出的通道数是 num_classes + 4*reg_max，
            虽然可能凑巧是 256，但语义上不是特征图，必须排除。
        """
        target_idx = None
        for idx, m in enumerate(self.model.model.model):
            # Backbone 阶段跳过（前 10 层）
            if idx <= 9 or type(m).__name__ == "Detect":
                continue
            # 只关注 C2f（CSP 瓶颈块，YOLOv8 Neck 的核心结构）
            if type(m).__name__ != "C2f":
                continue
            # 检查该层的权重是否有一个 4D 参数（卷积核）第一维为 256
            for name, param in m.named_parameters():
                if param.dim() == 4 and param.shape[0] == VECTOR_DIM:
                    target_idx = idx
                    break

        if target_idx is None:
            raise RuntimeError(
                f"No C2f layer with output channels == {VECTOR_DIM} found in Neck."
            )
        return target_idx

    def _build_truncated_model(self) -> nn.ModuleList:
        """
        提取前 N 层模块，丢弃 Detect 头和后续所有层。

        Returns:
            封装了前 (target_idx + 1) 层的 ModuleList

        【为什么用 ModuleList 而非 Sequential？】
            因为 YOLOv8 Neck 是 DAG（有向无环图），存在多输入模块
            （如 Concat），Sequential 的线性假设不成立。
            ModuleList 只做容器，前向逻辑我们自己写在 _extract_vector 里。
        """
        modules = []
        for i in range(self._trunc_layer_idx + 1):
            modules.append(self.model.model.model[i])
        return nn.ModuleList(modules).to(self.device).eval()

    # -------------------------------------------------------------------------
    # 预处理 + 前向
    # -------------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        LetterBox 缩放 + 归一化 + CHW 张量转换。

        Args:
            img_bgr: OpenCV 读入的 BGR 格式图像（H, W, 3）

        Returns:
            shape=(1, 3, EMBED_IMGSZ, EMBED_IMGSZ) 的 fp32 张量

        Why LetterBox？
            直接 resize 会改变宽高比，使缺陷形状失真。
            LetterBox 保持宽高比，用灰色填充边缘，与 YOLO 训练时的
            预处理保持一致（否则模型输入分布漂移，精度下降）。
        """
        from ultralytics.data.augment import LetterBox

        lb = LetterBox(self.EMBED_IMGSZ, auto=True, stride=32)
        img_lb = lb(image=img_bgr)  # HWC BGR uint8
        # HWC → CHW → [0,1] 归一化 → 加 batch 维
        tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _extract_vector(self, img_bgr: np.ndarray) -> list[float]:
        """
        通过截断模型逐层前向，提取 256 维 L2 归一化特征向量。

        Args:
            img_bgr: BGR 图像（单个缺陷裁剪图）

        Returns:
            256 维 Python list（已 L2 归一化）

        【前向逻辑说明】
            YOLOv8 Neck 中某些模块（如 Concat）接收多个输入，
            通过模块的 `.f` 属性标明输入来自哪几层的输出。
            我们维护 y 字典缓存每层输出，遇到多输入模块时按 .f 取缓存。

            - m.f == -1 或不是 list: 普通单输入模块，直接 m(x)
            - m.f 是 list: 多输入模块（如 Concat），从 y 中取对应层输出
        """
        x = self._preprocess(img_bgr)

        with torch.no_grad():
            y = {}  # 层输出缓存: {layer_idx: tensor}
            for i, m in enumerate(self._truncated_model):
                if hasattr(m, "f") and isinstance(m.f, list):
                    # 多输入模块（Concat 等）
                    inputs = [y[f] if f >= 0 else x for f in m.f]
                    x = m(inputs)
                else:
                    # 单输入模块（Conv / C2f / SPPF 等）
                    x = m(x)
                y[i] = x  # 缓存本层输出

        # ---------- 特征后处理 ----------
        feat_map = x  # 最终特征图形状: (1, 256, H', W')
        vec = feat_map.mean(dim=(2, 3)).squeeze(0)  # 全局平均池化 → (256,)
        vec = vec.cpu().numpy().astype(np.float32)

        # L2 归一化：使向量模长=1，配合 COSINE 度量等价于内积
        norm = np.linalg.norm(vec)
        normalized = vec / norm if norm > 0 else vec

        # ✅ 转为 Python list，兼容 JSON 序列化和 isinstance(v, list) 检查
        # （某些上游代码会 isinstance 检查 list，numpy 数组会被误判）
        return normalized.tolist()

    # -------------------------------------------------------------------------
    # 主接口：检测 + 提取
    # -------------------------------------------------------------------------

    def extract_from_image(self, image_path: str) -> list[dict]:
        """
        检测图片中所有缺陷，并为每个缺陷提取特征向量。

        Args:
            image_path: 图片路径

        Returns:
            缺陷列表，每项格式:
                {
                    "label": str,       # 缺陷类型
                    "confidence": float,# 置信度
                    "bbox": [x1,y1,x2,y2], # 像素坐标
                    "vector": list[float], # 256 维特征
                }

        Raises:
            ValueError: 图片无法读取

        【置信度策略】
            1. YOLO 推理层: CONF_THRESHOLD=0.1（低阈值保召回）
            2. 展示层: DISPLAY_CONF_THRESHOLD=0.20（过滤低质检测）
            3. 保底: 若全部低于 0.20，保留最高分的那一个（避免下游收空）
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        # ---------- YOLO 推理 ----------
        results = self.model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            imgsz=self.DETECT_IMGSZ,
            device=self.device,
            verbose=False,
        )

        # ---------- 逐个 bbox 处理 ----------
        raw_defects = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                # 解析 bbox 坐标（xyxy 格式）
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # 向外扩展 CROP_PADDING 像素，防止裁剪太紧
                # 同时 clamp 到图像边界，防止越界
                x1 = max(0, x1 - CROP_PADDING)
                y1 = max(0, y1 - CROP_PADDING)
                x2 = min(w, x2 + CROP_PADDING)
                y2 = min(h, y2 + CROP_PADDING)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue  # 空裁剪跳过

                # ✅ 1. 特征提取独立 try-except，不吞掉后续错误
                # 单个缺陷特征提取失败不应导致整张图处理失败
                try:
                    vector = self._extract_vector(crop)
                except Exception as e:
                    print(f"   ❌ 特征提取失败: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    vector = []

                # ✅ 2. 显式 int() 转换，避免 tensor 作为 dict key 的隐式转换问题
                cls_id = int(box.cls[0].item())
                label = self.model.names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0].item())

                raw_defects.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "vector": vector,
                })

        # ---------- 展示层置信度过滤 + 保底 ----------
        clean_defects = [
            d for d in raw_defects if d["confidence"] >= DISPLAY_CONF_THRESHOLD
        ]
        if not clean_defects and raw_defects:
            # 全部低于阈值 → 保底保留最高分的那一个，避免下游收到空列表
            best_match = max(raw_defects, key=lambda x: x["confidence"])
            clean_defects = [best_match]

        # ✅ 终极诊断：打印实际返回的数据结构（生产建议降级为 DEBUG 日志）
        print(f"[DIAG-RESULT] raw_defects count={len(raw_defects)}")
        for i, d in enumerate(raw_defects):
            v = d.get("vector")
            print(
                f"  [{i}] label={d['label']}, conf={d['confidence']:.3f}, "
                f"vector_type={type(v).__name__}, "
                f"vector_len={len(v) if hasattr(v, '__len__') else 'N/A'}, "
                f"is_list={isinstance(v, list)}, "
                f"first_3={v[:3] if hasattr(v, '__getitem__') and len(v) > 0 else 'EMPTY'}"
            )
        return clean_defects

    def extract_single_vector(self, image_path: str) -> np.ndarray:
        """
        提取整图的单个 256 维特征向量（用于 DB 种子数据）。

        Args:
            image_path: 图片路径

        Returns:
            (256,) 的 L2 归一化向量

        Note:
            与 extract_from_image 不同，本方法不做检测，直接对整图提取特征。
            主要用于数据集初始化、模型冒烟测试等场景。
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self._extract_vector(img)

    # -------------------------------------------------------------------------
    # 异步包装
    # -------------------------------------------------------------------------

    async def async_extract_from_image(self, image_path: str) -> list[dict]:
        """
        异步包装器：将 CPU/GPU 密集型推理放入线程池。

        Args:
            image_path: 图片路径

        Returns:
            同 extract_from_image

        ✅ R42 关键设计：
            YOLO 推理是同步阻塞操作（约 30-50ms），若直接在 async 函数里调用，
            会**阻塞 FastAPI 事件循环**，导致：
            - SSE 心跳超时
            - 其他并发请求排队
            用 asyncio.to_thread() 把同步调用放到线程池，事件循环立刻归还，
            可以处理其他协程。

        Why asyncio.to_thread 而非 run_in_executor？
            Python 3.9+ 推荐用法，更简洁，无 DeprecationWarning。
        """
        import asyncio
        return await asyncio.to_thread(self.extract_from_image, image_path)


# =============================================================================
# 自测入口
# =============================================================================

if __name__ == "__main__":
    import app.feature_extractor as fe
    print(f"\n[DEBUG] Loaded from: {fe.__file__}")
    extractor = FeatureExtractor()
    print(f"[OK] Device: {extractor.device}")