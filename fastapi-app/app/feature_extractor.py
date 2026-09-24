# -*- coding: utf-8 -*-
"""
YOLOv8 feature extractor for PCB defect vector retrieval.
Pipeline: detect defects -> crop each bbox -> truncated forward -> 256-dim L2-norm vector.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "runs" / "detect" / "runs" / "pcb_detect" / "train_v1" / "weights" / "best.pt")
)
VECTOR_DIM = 256
CROP_PADDING = 8
CONF_THRESHOLD = 0.1
DISPLAY_CONF_THRESHOLD = 0.20


class FeatureExtractor:
    DETECT_IMGSZ = 1280
    EMBED_IMGSZ = 640

    def __init__(self, model_path: str = str(MODEL_PATH)):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        os.environ["TORCH_FORCE_WEIGHTS_ONLY"] = "0"
        try:
            self.model = YOLO(model_path)
        finally:
            os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY", None)

        self.device = "cpu"
        self.model.to(self.device)
        self.model.model.float()

        # ✅ 构建截断模型：只保留到目标 C2f 层
        self._trunc_layer_idx = self._find_neck_layer()
        self._truncated_model = self._build_truncated_model()
        print(f"[INIT] Truncated model built. Stops at layer {self._trunc_layer_idx} (C2f, {VECTOR_DIM}-ch)")

    def _find_neck_layer(self) -> int:
        """Locate the last C2f layer in Neck with output channels == VECTOR_DIM."""
        target_idx = None
        for idx, m in enumerate(self.model.model.model):
            if idx <= 9 or type(m).__name__ == "Detect":
                continue
            if type(m).__name__ != "C2f":
                continue
            for name, param in m.named_parameters():
                if param.dim() == 4 and param.shape[0] == VECTOR_DIM:
                    target_idx = idx
                    break
        if target_idx is None:
            raise RuntimeError(f"No C2f layer with output channels == {VECTOR_DIM} found in Neck.")
        return target_idx

    def _build_truncated_model(self) -> nn.ModuleList:
        """提取前 N 层模块，丢弃 Detect 头和后续所有层"""
        modules = []
        for i in range(self._trunc_layer_idx + 1):
            modules.append(self.model.model.model[i])
        return nn.ModuleList(modules).to(self.device).eval()

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """LetterBox + Normalize + CHW tensor"""
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(self.EMBED_IMGSZ, auto=True, stride=32)
        img_lb = lb(image=img_bgr)
        tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _extract_vector(self, img_bgr: np.ndarray) -> np.ndarray:
        """通过截断模型逐层前向传播提取特征"""
        x = self._preprocess(img_bgr)

        with torch.no_grad():
            y = {}
            for i, m in enumerate(self._truncated_model):
                if hasattr(m, 'f') and isinstance(m.f, list):
                    inputs = [y[f] if f >= 0 else x for f in m.f]
                    x = m(inputs)
                else:
                    x = m(x)
                y[i] = x

        feat_map = x
        vec = feat_map.mean(dim=(2, 3)).squeeze(0)
        vec = vec.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        normalized = vec / norm if norm > 0 else vec
        
        # ✅ 转为 Python list，兼容 JSON 序列化和 isinstance(v, list) 检查
        return normalized.tolist()

    def extract_from_image(self, image_path: str) -> list[dict]:
        """Detect all defects and extract one 256-dim vector per defect."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        results = self.model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            imgsz=self.DETECT_IMGSZ,
            device=self.device,
            verbose=False
        )

        raw_defects = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1 - CROP_PADDING)
                y1 = max(0, y1 - CROP_PADDING)
                x2 = min(w, x2 + CROP_PADDING)
                y2 = min(h, y2 + CROP_PADDING)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # ✅ 1. 特征提取独立 try-except，不再吞掉后续错误
                try:
                    vector = self._extract_vector(crop)
                except Exception as e:
                    print(f"   ❌ 特征提取失败: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
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

        clean_defects = [d for d in raw_defects if d["confidence"] >= DISPLAY_CONF_THRESHOLD]
        if not clean_defects and raw_defects:
            best_match = max(raw_defects, key=lambda x: x["confidence"])
            clean_defects = [best_match]
        # ✅ 终极诊断：打印实际返回的数据结构
        print(f"[DIAG-RESULT] raw_defects count={len(raw_defects)}")
        for i, d in enumerate(raw_defects):
            v = d.get("vector")
            print(f"  [{i}] label={d['label']}, conf={d['confidence']:.3f}, "
                  f"vector_type={type(v).__name__}, "
                  f"vector_len={len(v) if hasattr(v, '__len__') else 'N/A'}, "
                  f"is_list={isinstance(v, list)}, "
                  f"first_3={v[:3] if hasattr(v, '__getitem__') and len(v) > 0 else 'EMPTY'}")
        return clean_defects

    def extract_single_vector(self, image_path: str) -> np.ndarray:
        """Extract one 256-dim vector for the whole image (DB seeding)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self._extract_vector(img)

    async def async_extract_from_image(self, image_path: str) -> list[dict]:
        """
        ✅ R42：异步包装器，将 GPU/CPU 密集型推理放入线程池
        避免阻塞 FastAPI 事件循环，保证 SSE 心跳和其他并发请求不受影响
        """
        import asyncio
        # ✅ 使用 Python 3.9+ 推荐的 to_thread，更简洁且无 DeprecationWarning
        return await asyncio.to_thread(self.extract_from_image, image_path)


if __name__ == "__main__":
    import app.feature_extractor as fe
    print(f"\n[DEBUG] Loaded from: {fe.__file__}")
    extractor = FeatureExtractor()
    print(f"[OK] Device: {extractor.device}")