from __future__ import annotations

import logging
from typing import List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

Track = Tuple[int, int, int, int, int]
Detection = Tuple[int, int, int, int, float]


def _iou(boxA, boxB) -> float:
    # boxes as (l,t,w,h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    return float(interArea) / union if union > 0 else 0.0


class _SimpleIOUTracker:
    """Very small IOU tracker to provide stable IDs when DeepSORT fails.

    Tracks are represented as dicts with `id` and `bbox`=(l,t,w,h). Matching is
    greedy by highest IOU > threshold.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        det_boxes = [ (x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, score) in detections ]
        updated = {}

        # match existing tracks
        for tid, t in self.tracks.items():
            best_iou = 0.0
            best_idx = None
            for i, db in enumerate(det_boxes):
                iou_score = _iou(t["bbox"], db)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_idx = i
            if best_idx is not None and best_iou >= self.iou_threshold:
                updated[tid] = {"bbox": det_boxes[best_idx], "lost": 0}
                det_boxes[best_idx] = None
            else:
                # not matched
                t["lost"] = t.get("lost", 0) + 1
                if t["lost"] <= self.max_lost:
                    updated[tid] = t

        # create new tracks for unmatched detections
        for db in det_boxes:
            if db is None:
                continue
            tid = self._next_id
            self._next_id += 1
            updated[tid] = {"bbox": db, "lost": 0}

        self.tracks = updated

        # return tracks in (track_id, x1, y1, x2, y2) format
        out = []
        for tid, info in self.tracks.items():
            l, t, w, h = info["bbox"]
            out.append((tid, int(l), int(t), int(l + w), int(t + h)))
        return out


class PersonTracker:
    """Deep SORT based tracker for person detections."""

    def __init__(self) -> None:
        try:
            self.tracker = DeepSort(max_age=30)
            logger.info("Initialized DeepSort tracker")
        except Exception:
            logger.exception("Failed to initialize DeepSort; will use IOU fallback")
            self.tracker = None
        # lightweight IOU tracker as a fallback when DeepSort fails
        self._simple_tracker = _SimpleIOUTracker()

    def update(self, detections: List[Detection], frame=None) -> List[Track]:
        """Update tracker with detections.

        Args:
            detections: List of detections from detector.

        Returns:
            List of tracks as (track_id, x1, y1, x2, y2).
        """

        # Build DeepSort raw_detections format: ([l,t,w,h], conf, class)
        tracker_inputs = [
            ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(score), None)
            for (x1, y1, x2, y2, score) in detections
        ]

        # If DeepSort isn't available, or there are no detections, use simple IOU tracker
        if self.tracker is None or len(tracker_inputs) == 0:
            return self._simple_tracker.update(detections)

        embeds = None
        if frame is not None and getattr(self.tracker, "embedder", None) is not None:
            crops = []
            im_h, im_w = frame.shape[:2]
            for raw_det in tracker_inputs:
                bbox = raw_det[0]
                l, t, w, h = map(int, bbox)
                r = l + w
                b = t + h
                l_c = max(0, l)
                t_c = max(0, t)
                r_c = min(im_w, r)
                b_c = min(im_h, b)
                if r_c <= l_c or b_c <= t_c:
                    crops.append(frame[0:1, 0:1])
                else:
                    crops.append(frame[t_c:b_c, l_c:r_c])
            try:
                embeds = self.tracker.embedder.predict(crops)
            except Exception:
                logger.exception("Embedder failed; proceeding without embeddings")

        try:
            tracked_objects = self.tracker.update_tracks(tracker_inputs, embeds=embeds, frame=frame)
        except Exception:
            logger.exception("DeepSort update_tracks failed; falling back to simple IOU tracker")
            return self._simple_tracker.update(detections)

        # Convert DeepSort track objects to our expected (id,x1,y1,x2,y2) format
        out: List[Track] = []
        for tr in tracked_objects:
            try:
                if not getattr(tr, "is_confirmed", lambda: True)():
                    continue
                track_id = tr.track_id
                try:
                    track_id = int(track_id)
                except Exception:
                    pass
                if hasattr(tr, "to_ltrb"):
                    l, t, r, b = tr.to_ltrb()
                elif hasattr(tr, "to_tlbr"):
                    l, t, r, b = tr.to_tlbr()
                else:
                    # fallback: try original_ltwh if available
                    ltwh = getattr(tr, "original_ltwh", None)
                    if ltwh is None:
                        continue
                    l, t, w, h = ltwh
                    r = l + w
                    b = t + h
                out.append((track_id, int(l), int(t), int(r), int(b)))
            except Exception:
                logger.exception("Error converting track object; skipping")
        return out
