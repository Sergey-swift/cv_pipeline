import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Sequence, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # ⬅️ Ensure ultralytics>=0.4.0 with YOLOv11 weights

# ---------------------------------------------------------------------------
# Logger configuration (stdout). Upstream apps can override handlers/levels.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: Letterbox resize (stride-32 friendly, like Ultralytics impl.)
# ---------------------------------------------------------------------------

def _letterbox(
    img: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize *img* to *new_shape* with unchanged aspect ratio using padding."""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    r = min(new_w / shape[1], new_h / shape[0])
    new_unpadded = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_w - new_unpadded[0], new_h - new_unpadded[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpadded, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


# ---------------------------------------------------------------------------
# Core Detector
# ---------------------------------------------------------------------------

class YOLODetector:
    """Modular wrapper around Ultralytics **YOLOv11** for scalable CV pipelines."""

    def __init__(
        self,
        model_path: Union[str, Path] | None = None,
        device: str | torch.device | None = None,
        class_names: Optional[Sequence[str]] = None,
        confidence_thresholds: Optional[Dict[int, float]] = None,
        input_size: int = 640,
        use_fp16: bool = True,                          # ← renamed for clarity / parity with other models
    ) -> None:
        # ------------------------------------------------------------------
        # Device & model loading
        # ------------------------------------------------------------------
        self.device: torch.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model_path is None:
            model_path = Path("models/weights/yolo11n.pt")  # or s/m/l/x depending on what you downloaded
            logger.warning(f"[YOLODetector] No *model_path* provided – falling back to '{model_path}'.")


        logger.info(f"Loading YOLOv11 ➜ '{model_path}' on {self.device}")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        # Optional FP16 for CUDA (guarded)
        self.fp16: bool = use_fp16 and self.device.type == "cuda"
        if self.fp16:
            logger.info("Using FP16 inference")
            self.model.model.half()

        # Metadata
        self.class_names = list(class_names) if class_names is not None else self.model.names
        self.confidence_thresholds = confidence_thresholds or {}
        self.input_size = input_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        images: Union[str, Path, np.ndarray, List[np.ndarray], List[str], List[Path]],
        batch_size: int = 1,
        iou_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        **predict_kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """Run object detection on *images*.

        Parameters
        ----------
        images : Union[path/str/ndarray, list]
            Single image or sequence of images/paths.
        batch_size : int
            Batch size for inference.
        iou_threshold : float, optional
            IoU threshold for NMS (passed to Ultralytics as ``iou``).
        max_det : int, optional
            Maximum detections per image (passed as ``max_det``).
        predict_kwargs : dict
            Additional keyword arguments forwarded to ``YOLO.__call__``.
        """
        # Map custom NMS parameters
        if iou_threshold is not None:
            predict_kwargs.setdefault("iou", iou_threshold)
        if max_det is not None:
            predict_kwargs.setdefault("max_det", max_det)

        # Normalise input to list
        if not isinstance(images, (list, tuple)):
            images = [images]

        # Preprocess
        tensors, metas = [], []
        for item in images:
            img = self._load_image(item)
            t, r, pad = self._preprocess(img)
            tensors.append(t)
            metas.append((img.shape[:2], r, pad))

        batch_tensor = torch.stack(tensors, 0).to(self.device, non_blocking=True)
        if self.fp16:
            batch_tensor = batch_tensor.half()

        # Inference
        with torch.no_grad():
            results = self.model(batch_tensor, batch=batch_size, verbose=False, **predict_kwargs)

        # Post-process per-image
        all_dets: List[List[Dict[str, Any]]] = []
        for res, meta in zip(results, metas):
            all_dets.append(self._postprocess(res, meta))
        return all_dets

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        class_names: Optional[Sequence[str]] = None,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Return *image* annotated with bounding boxes and labels.

        The function validates bbox keys to avoid crashes if structure is wrong.
        """
        annot = image.copy()
        for det in detections:
            if not (
                isinstance(det, dict)
                and "bbox" in det
                and all(k in det["bbox"] for k in ("x1", "y1", "x2", "y2"))
            ):
                logging.warning("[YOLODetector] Skipping malformed detection entry: %s", det)
                continue

            x1, y1, x2, y2 = map(int, det["bbox"].values())
            conf = det.get("confidence", 0.0)
            cid = det.get("class_id", -1)
            label = (
                class_names[cid] if class_names is not None and 0 <= cid < len(class_names) else str(cid)
            )
            caption = f"{label} {conf:.2f}"

            cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
            # Label background
            t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annot, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annot, caption, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return annot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(src: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(src, (str, Path)):
            path = str(src)
            if not Path(path).exists():
                raise FileNotFoundError(path)
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
        elif isinstance(src, np.ndarray):
            img = src
        else:
            raise TypeError("Unsupported image input type")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lb, r, (pad_w, pad_h) = _letterbox(img_rgb, self.input_size)
        tensor = torch.as_tensor(lb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return tensor, r, (pad_w, pad_h)

    def _postprocess(
        self,
        yolo_result: Any,  # ultralytics.engine.results.Results
        meta: Tuple[Tuple[int, int], float, Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        orig_shape, r, (pad_w, pad_h) = meta

        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return []

        boxes = yolo_result.boxes.xyxy.cpu().numpy()  # (n, 4) letterbox coords
        scores = yolo_result.boxes.conf.cpu().numpy()
        cids = yolo_result.boxes.cls.cpu().numpy().astype(int)

        # Vectorised confidence filtering (no np.vectorize)
        thresh_arr = np.array([self.confidence_thresholds.get(cid, 0.25) for cid in cids])
        mask = scores >= thresh_arr
        boxes, scores, cids = boxes[mask], scores[mask], cids[mask]

        # Reverse letterbox to original coords
        if boxes.size:
            boxes[:, [0, 2]] -= pad_w
            boxes[:, [1, 3]] -= pad_h
            boxes /= r
            boxes = boxes.clip(0, None)

        detections: List[Dict[str, Any]] = []
        for bb, sc, cid in zip(boxes, scores, cids):
            x1, y1, x2, y2 = bb.tolist()
            det = {
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": float(sc),
                "class_id": int(cid),
            }
            if self.class_names is not None and 0 <= cid < len(self.class_names):
                det["class_label"] = self.class_names[cid]
            detections.append(det)
        return detections


# ---------------------------------------------------------------------------
# Quick test (comment out in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11 quick test")
    parser.add_argument("image", help="Path to image for detection")
    parser.add_argument("--weights", default="yolov11.pt", help="YOLOv11 weights path")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    args = parser.parse_args()

    detector = YOLODetector(model_path=args.weights)
    detections = detector.detect(args.image, iou_threshold=args.iou, max_det=args.max_det)[0]

    logger.info("Detections:\n" + "\n".join(map(str, detections)))

    img_bgr = cv2.imread(args.image)
    vis = YOLODetector.draw_boxes(img_bgr, detections, detector.class_names)
    cv2.imshow("result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
