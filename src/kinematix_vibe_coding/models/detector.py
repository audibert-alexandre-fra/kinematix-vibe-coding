"""Détection de personne (RF-DETR, Apache-2.0) et suivi de la personne principale."""

from __future__ import annotations

import numpy as np

COCO_PERSON_CLASS_ID = 1  # RF-DETR expose les identifiants COCO (1-indexés)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class RFDETRDetector:
    """Détecteur de personnes RF-DETR. Variantes nano/small/medium/base : licence Apache-2.0."""

    _VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase"}

    def __init__(self, variant: str = "medium", confidence: float = 0.4):
        import rfdetr

        if variant not in self._VARIANTS:
            raise ValueError(f"Variante RF-DETR inconnue {variant!r} ({list(self._VARIANTS)})")
        self._model = getattr(rfdetr, self._VARIANTS[variant])()
        if hasattr(self._model, "optimize_for_inference"):
            self._model.optimize_for_inference()
        self.confidence = confidence
        self.variant = variant

    def detect(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Renvoie [(bbox xyxy, score)] des personnes, triées par score décroissant."""
        from PIL import Image

        dets = self._model.predict(Image.fromarray(image_rgb), threshold=self.confidence)
        out = []
        if len(dets) > 0 and dets.class_id is not None:
            mask = dets.class_id == COCO_PERSON_CLASS_ID
            for box, score in zip(dets.xyxy[mask], dets.confidence[mask], strict=True):
                out.append((np.asarray(box, dtype=np.float32), float(score)))
        out.sort(key=lambda d: -d[1])
        return out


class MainPersonTracker:
    """Suit une seule personne : la plus grande/centrée au départ, puis continuité par IoU.

    Si aucune détection ne correspond, la boîte précédente est conservée (avec un compteur
    de pertes) pour ne pas sauter vers une autre personne."""

    def __init__(self, image_size: tuple[int, int], min_iou: float = 0.2, max_lost: int = 15):
        self.w, self.h = image_size
        self.min_iou = min_iou
        self.max_lost = max_lost
        self.box: np.ndarray | None = None
        self.lost = 0

    def _initial_score(self, box: np.ndarray, score: float) -> float:
        area = (box[2] - box[0]) * (box[3] - box[1]) / (self.w * self.h)
        cx, cy = (box[0] + box[2]) / 2 / self.w, (box[1] + box[3]) / 2 / self.h
        centered = 1.0 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) * 2)
        return area * (0.5 + centered) * score

    def update(self, detections: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray | None, bool]:
        """Renvoie (bbox, détectée_cette_image)."""
        if self.box is None:
            if not detections:
                return None, False
            self.box = max(detections, key=lambda d: self._initial_score(*d))[0]
            return self.box, True
        best, best_iou = None, self.min_iou
        for box, _ in detections:
            v = iou(box, self.box)
            if v > best_iou:
                best, best_iou = box, v
        if best is None:
            self.lost += 1
            if self.lost > self.max_lost and detections:  # ré-initialisation
                self.box, self.lost = None, 0
                return self.update(detections)
            return self.box, False
        self.box, self.lost = best, 0
        return best, True
