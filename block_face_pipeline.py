"""
BlockFacePipeline — Python 版本，基于 Ultralytics YOLO。
与原 C++ ONNX 管线逻辑一致，三个模型分别做检测、分割、分类。
用法：直接运行脚本，或导入 BlockFacePipeline 类。
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 几何工具（与 C++ geometry_utils.hpp 等价）
# ---------------------------------------------------------------------------

def order_points(pts):
    """四点排序：[左上, 右上, 右下, 左下]"""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def warp_polygon_to_square(img, poly, output_size=512):
    """将四边形区域透视矫正为正方形"""
    src = order_points(poly)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (output_size, output_size))


def approx_to_quad(contour):
    """轮廓逼近为四边形，失败则回退到 minAreaRect"""
    peri = cv2.arcLength(contour, True)
    for factor in (0.02, 0.04, 0.08, 0.12):
        approx = cv2.approxPolyDP(contour, factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect)


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------

class BlockFacePipeline:
    def __init__(
        self,
        detect_model_path: str,
        detect_classes: list,
        segment_model_path: str,
        seg_classes: list,
        classify_model_path: str,
        face_classes: list,
        warp_size: int = 512,
        detect_conf: float = 0.5,
        seg_conf: float = 0.3,
        min_face_area: float = 500,
    ):
        self.detector = YOLO(detect_model_path)
        self.segmentor = YOLO(segment_model_path)
        self.classifier = YOLO(classify_model_path)

        self.detect_classes = detect_classes
        self.seg_classes = seg_classes
        self.face_classes = face_classes
        self.warp_size = warp_size
        self.detect_conf = detect_conf
        self.seg_conf = seg_conf
        self.min_face_area = min_face_area

    # ---- Stage 1: 检测方块 -------------------------------------------------

    def detect_blocks(self, image):
        """返回 list[dict]: box(x,y,w,h), class_id, class_name, confidence"""
        results = self.detector(image, conf=self.detect_conf, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                detections.append({
                    "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "class_id": cls,
                    "class_name": self.detect_classes[cls] if cls < len(self.detect_classes) else str(cls),
                    "confidence": float(conf),
                })
        return detections

    def draw_detections(self, image, detections):
        """在原图上绘制检测框（与 C++ 版本一致）"""
        for det in detections:
            x, y, w, h = det["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- 辅助：裁剪方块区域 -------------------------------------------------

    def crop_block(self, image, box, pad=10):
        """从原图中裁出方块区域（带 padding）"""
        x, y, w, h = box
        h_img, w_img = image.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    # ---- Stage 2: 分割面 ---------------------------------------------------

    def extract_instances(self, image):
        """返回 list[dict]: binary_mask, bbox(x,y,w,h), class_id, class_name, score"""
        results = self.segmentor(image, conf=self.seg_conf, verbose=False)[0]
        faces = []
        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()       # (N, H, W) float [0,1]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            h, w = image.shape[:2]
            for mask_arr, box, conf, cls in zip(masks_data, boxes, confs, clss):
                # mask_arr 已缩放到原图尺寸
                if mask_arr.shape != (h, w):
                    mask_arr = cv2.resize(mask_arr, (w, h))
                binary = (mask_arr > 0.5).astype(np.uint8) * 255
                x1, y1, x2, y2 = box
                faces.append({
                    "binary_mask": binary,
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "class_id": cls,
                    "class_name": self.seg_classes[cls] if cls < len(self.seg_classes) else str(cls),
                    "score": float(conf),
                })
        return faces

    def draw_instances(self, image, faces):
        """在图像上绘制分割轮廓和标签（与 C++ 版本一致）"""
        for face in faces:
            contours, _ = cv2.findContours(face["binary_mask"], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            biggest = max(contours, key=cv2.contourArea)
            cv2.polylines(image, [biggest], True, (0, 255, 0), 2)
            bx, by = face["bbox"][:2]
            label = f"{face['class_name']}: {face['score']:.2f}"
            cv2.putText(image, label, (bx, max(0, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- Stage 3: 分类面 ---------------------------------------------------

    def classify_face(self, cropped_image, face, source_box=None, face_index=0):
        """对单张矫正后的面分类，返回 FaceResult dict"""
        result = {
            "warped_face": None,
            "class_id": -1,
            "class_name": "",
            "confidence": 0.0,
            "source_bbox": source_box or face.get("bbox"),
            "face_index": face_index,
        }

        # 1. 从二值掩码找轮廓
        binary_mask = face["binary_mask"]
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result

        # 2. 最大轮廓 & 面积阈值
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_face_area:
            return result

        # 3. 四边形近似
        quad = approx_to_quad(biggest)
        if len(quad) != 4:
            return result

        # 4. 透视矫正
        warped = warp_polygon_to_square(cropped_image, quad, self.warp_size)

        # 5. 分类
        cls_results = self.classifier(warped, verbose=False)[0]
        class_id = cls_results.probs.top1
        confidence = float(cls_results.probs.top1conf)

        result["warped_face"] = warped
        result["class_id"] = class_id
        result["class_name"] = (
            self.face_classes[class_id]
            if class_id < len(self.face_classes)
            else str(class_id)
        )
        result["confidence"] = confidence
        return result


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

def main():
    pipeline = BlockFacePipeline(
        detect_model_path="model/block_detect.pt",
        detect_classes=["red", "blue", "empty"],
        segment_model_path="model/block_seg.pt",
        seg_classes=["red_face", "blue_face"],
        classify_model_path="model/block_cls.pt",
        face_classes=["falsekfs", "r1kfs", "r2kfs", "unknown"],
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: 检测
        blocks = pipeline.detect_blocks(frame)
        pipeline.draw_detections(frame, blocks)

        for block in blocks:
            cropped = pipeline.crop_block(frame, block["box"])
            if cropped is None:
                continue

            # Stage 2: 分割
            faces = pipeline.extract_instances(cropped)
            display = cropped.copy()
            pipeline.draw_instances(display, faces)
            cv2.imshow("Cropped Block", display)

            # Stage 3: 分类
            for i, face in enumerate(faces):
                cls_result = pipeline.classify_face(cropped, face, face["bbox"], i)
                if cls_result["class_id"] >= 0:
                    print(
                        f"Detected face: class={cls_result['class_name']}, "
                        f"confidence={cls_result['confidence']:.3f}"
                    )
                    cv2.imshow("Warped Face", cls_result["warped_face"])

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
