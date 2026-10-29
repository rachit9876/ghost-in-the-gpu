"""Ghost Face Mirror — 100% GPU Face Detection, Pose/Occlusion Quality Gate & ArcFace Extractor.
Executes SCRFD 10G + 5-Point Similarity Alignment + Pose/Visibility Gate + ArcFace R50 on CUDA.
"""
import warnings
import cv2
import numpy as np
import torch

from face_mirror.config import (
    SCRFD_PT_PATH, ARCFACE_PT_PATH, DEFAULT_THRESHOLDS, ensure_gpu_models
)

warnings.filterwarnings("ignore", category=UserWarning)

# Standard ArcFace 112x112 5-point reference landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class GPUFaceEngine:
    def __init__(self, device: str = None):
        ensure_gpu_models()
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for Ghost Face Mirror, but torch.cuda.is_available() is False!")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(device)

        self.gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        self.det_model = torch.load(SCRFD_PT_PATH, map_location=self.device, weights_only=False).eval().to(self.device)
        self.rec_model = torch.load(ARCFACE_PT_PATH, map_location=self.device, weights_only=False).eval().to(self.device)

    def detect_raw(self, img_bgr: np.ndarray, input_size=(640, 640), score_thresh=0.25):
        """Runs SCRFD 10G on GPU and returns all raw candidate faces before quality filtering."""
        if img_bgr is None or img_bgr.size == 0:
            return []

        im_ratio = float(img_bgr.shape[0]) / img_bgr.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img_bgr.shape[0]
        resized_img = cv2.resize(img_bgr, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        tensor_in = torch.from_numpy(blob).to(self.device, non_blocking=True)
        with torch.no_grad():
            net_outs = self.det_model(tensor_in)
        outs = [o.detach().cpu().numpy() for o in net_outs]

        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2
        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(feat_stride_fpn):
            scores = outs[idx]
            bbox_preds = outs[idx + fmc] * stride
            kps_preds = outs[idx + fmc * 2] * stride
            if scores.ndim == 3:
                scores, bbox_preds, kps_preds = scores[0], bbox_preds[0], kps_preds[0]

            height, width = input_size[1] // stride, input_size[0] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores[:, 0] >= score_thresh)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = scores[pos_inds, 0]
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_centers = anchor_centers[pos_inds]

            bboxes = np.stack([
                pos_centers[:, 0] - pos_bbox_preds[:, 0],
                pos_centers[:, 1] - pos_bbox_preds[:, 1],
                pos_centers[:, 0] + pos_bbox_preds[:, 2],
                pos_centers[:, 1] + pos_bbox_preds[:, 3]
            ], axis=-1)

            pos_kps_preds = kps_preds[pos_inds]
            kpss = []
            for i in range(0, pos_kps_preds.shape[1], 2):
                px = pos_centers[:, 0] + pos_kps_preds[:, i]
                py = pos_centers[:, 1] + pos_kps_preds[:, i + 1]
                kpss.append(np.stack([px, py], axis=-1))
            kpss = np.stack(kpss, axis=1)

            scores_list.append(pos_scores)
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        if not scores_list:
            return []

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
        kpss = np.concatenate(kpss_list, axis=0) / det_scale

        # NMS
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
            yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
            xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
            yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            area_i = (bboxes[i, 2] - bboxes[i, 0] + 1) * (bboxes[i, 3] - bboxes[i, 1] + 1)
            area_o = (bboxes[order[1:], 2] - bboxes[order[1:], 0] + 1) * (bboxes[order[1:], 3] - bboxes[order[1:], 1] + 1)
            ovr = inter / (area_i + area_o - inter)
            inds = np.where(ovr <= 0.4)[0]
            order = order[inds + 1]

        return [
            {
                "score": float(scores[k]),
                "bbox": bboxes[k].astype(np.float32),
                "kps": kpss[k].astype(np.float32),
            }
            for k in keep
        ]

    def evaluate_face_quality(self, img_bgr: np.ndarray, face: dict, thresholds: dict = None) -> dict:
        """Evaluates 5-point landmark visibility (eyes, nose, lips) and 3D pose angle.
        Returns a detailed dictionary with pass/fail boolean, rejection reason, metrics, and aligned 112x112 crop.
        """
        th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        score = face["score"]
        bbox = face["bbox"]
        kps = face["kps"]  # (5, 2): [left_eye, right_eye, nose, left_mouth, right_mouth]

        box_w = float(bbox[2] - bbox[0])
        box_h = float(bbox[3] - bbox[1])

        # 1. Detection Confidence Gate (visibility of eyes/nose/mouth)
        if score < th["min_det_score"]:
            return {
                "passed": False,
                "reason": f"low_visibility_confidence ({score:.2f} < {th['min_det_score']:.2f})",
                "score": score,
                "yaw_asymmetry": 0.0,
                "align_error": 0.0,
                "mouth_grad_ratio": 0.0,
            }

        # 2. Minimum Face Resolution Gate
        if box_w < th["min_face_size"] or box_h < th["min_face_size"]:
            return {
                "passed": False,
                "reason": f"face_too_small ({int(box_w)}x{int(box_h)}px < {int(th['min_face_size'])}px)",
                "score": score,
                "yaw_asymmetry": 0.0,
                "align_error": 0.0,
                "mouth_grad_ratio": 0.0,
            }

        # 3. 3D Yaw / Side-Angle Gate (projection of nose onto inter-ocular axis)
        le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_vec = re - le
        eye_len = float(np.linalg.norm(eye_vec)) + 1e-6
        eye_unit = eye_vec / eye_len

        # Nose projection along inter-ocular vector: 0.5 = perfectly frontal center; <0 or >1 = profile side view
        nose_proj = float(np.dot(nose - le, eye_unit)) / eye_len
        yaw_asymmetry = abs(nose_proj - 0.5) * 2.0

        # Estimate similarity transform to canonical 112x112 ArcFace coordinates
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_SRC, method=cv2.LMEDS)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_SRC)
        if M is None:
            return {
                "passed": False,
                "reason": "degenerate_landmarks",
                "score": score,
                "yaw_asymmetry": yaw_asymmetry,
                "align_error": 99.0,
                "mouth_grad_ratio": 0.0,
            }

        kps_homo = np.hstack([kps, np.ones((5, 1), dtype=np.float32)])
        kps_aligned = (M @ kps_homo.T).T
        align_error = float(np.mean(np.linalg.norm(kps_aligned - ARCFACE_SRC, axis=1)))

        # Warp to 112x112 and measure lower-face (lips/mouth) vs eye horizontal edge energy
        warped = cv2.warpAffine(img_bgr, M, (112, 112))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        eye_grad = float(np.mean(sobel_y[40:62, 25:87]))
        mouth_grad = float(np.mean(sobel_y[82:102, 35:77]))
        mouth_grad_ratio = mouth_grad / (eye_grad + 1e-5)

        if yaw_asymmetry > th["max_yaw_asymmetry"]:
            return {
                "passed": False,
                "reason": f"extreme_side_angle (yaw={yaw_asymmetry:.2f} > {th['max_yaw_asymmetry']:.2f})",
                "score": score,
                "yaw_asymmetry": yaw_asymmetry,
                "align_error": align_error,
                "mouth_grad_ratio": mouth_grad_ratio,
            }

        if align_error > th["max_align_error"]:
            return {
                "passed": False,
                "reason": f"distorted_pose_geometry (err={align_error:.2f} > {th['max_align_error']:.2f})",
                "score": score,
                "yaw_asymmetry": yaw_asymmetry,
                "align_error": align_error,
                "mouth_grad_ratio": mouth_grad_ratio,
            }

        # 4. Lower-Face / Mouth Visibility Gate (eliminates occluded mouths like Image #4)
        if mouth_grad_ratio < th["min_mouth_grad_ratio"]:
            return {
                "passed": False,
                "reason": f"occluded_mouth_or_lips (ratio={mouth_grad_ratio:.2f} < {th['min_mouth_grad_ratio']:.2f})",
                "score": score,
                "yaw_asymmetry": yaw_asymmetry,
                "align_error": align_error,
                "mouth_grad_ratio": mouth_grad_ratio,
            }

        return {
            "passed": True,
            "reason": "valid_frontal_face",
            "score": score,
            "yaw_asymmetry": yaw_asymmetry,
            "align_error": align_error,
            "mouth_grad_ratio": mouth_grad_ratio,
            "bbox": bbox.tolist(),
            "kps": kps.tolist(),
            "warped_bgr": warped,
        }

    def extract_embedding(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Runs ArcFace R50 on GPU for an aligned 112x112 BGR face crop and returns L2-normalized 512D vector."""
        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, 112, 112)
        tensor_in = torch.from_numpy(chw).to(self.device, non_blocking=True)
        with torch.no_grad():
            feat = self.rec_model(tensor_in)
        vec = feat.detach().cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def process_image(self, img_bgr: np.ndarray, thresholds: dict = None) -> dict:
        """Detects faces, runs strict visibility & angle gates, and extracts ArcFace on the best passing face."""
        raw_faces = self.detect_raw(img_bgr)
        if not raw_faces:
            return {
                "status": "rejected",
                "reason": "no_face_detected",
                "faces_found": 0,
                "embedding": None,
                "metrics": None,
            }

        evals = []
        for rf in raw_faces:
            ev = self.evaluate_face_quality(img_bgr, rf, thresholds=thresholds)
            evals.append(ev)

        passed_faces = [e for e in evals if e["passed"]]
        if not passed_faces:
            # Return the primary face's rejection reason
            best_fail = evals[0]
            return {
                "status": "rejected",
                "reason": best_fail["reason"],
                "faces_found": len(raw_faces),
                "embedding": None,
                "metrics": {
                    "score": round(best_fail["score"], 3),
                    "yaw_asymmetry": round(best_fail["yaw_asymmetry"], 3),
                    "align_error": round(best_fail["align_error"], 3),
                    "mouth_grad_ratio": round(best_fail["mouth_grad_ratio"], 3),
                },
            }

        # Pick the highest-confidence passing face (or largest area)
        best = max(passed_faces, key=lambda x: x["score"])
        emb = self.extract_embedding(best["warped_bgr"])
        return {
            "status": "accepted",
            "reason": "valid_frontal_face",
            "faces_found": len(raw_faces),
            "embedding": emb,
            "bbox": best["bbox"],
            "kps": best["kps"],
            "metrics": {
                "score": round(best["score"], 3),
                "yaw_asymmetry": round(best["yaw_asymmetry"], 3),
                "align_error": round(best["align_error"], 3),
                "mouth_grad_ratio": round(best["mouth_grad_ratio"], 3),
            },
        }
