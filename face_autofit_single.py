import numpy as np
import torch
import cv2
import math
import os
import time  # 添加time模块导入
from PIL import Image
import logging
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceAutoFit")

# 添加性能计时器
class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        if elapsed > 0.1:  # 只记录超过0.1秒的操作
            logger.info(f"性能计时 - {self.name}: {elapsed:.3f}秒")

class FaceAutoFitSingle:
    """智能人脸适配节点 - 优化版"""
    
    # 类变量用于缓存模型
    _face_cascade = None
    _mediapipe_face_mesh = None
    _dlib_detector = None
    _dlib_predictor = None
    
    # 添加缓存变量
    _last_image_hash = None
    _last_face_info = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "face_mask": ("MASK",),
                "mode": (["portrait", "half_body", "full_body", "custom"], {"default": "portrait"}),
                "background": (["white", "gray"], {"default": "white"}),
                "move_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "move_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "face_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "detection_method": (["auto", "opencv", "mediapipe", "dlib", "none"], {"default": "dlib"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "image/face"
    FUNCTION = "process"

    def process(self, face_image, face_mask, mode, background, 
               move_x, move_y, face_size, angle=0.0, detection_method="dlib", 
               custom_width=1024, custom_height=1024):
        try:
            with Timer("总处理时间"):
                # 保存当前模式和检测方法，供其他方法使用
                self.current_mode = mode
                self.detection_method = detection_method
                
                # 计算目标尺寸和背景颜色
                target_w, target_h = self._calculate_dimensions(mode, custom_width, custom_height)
                bg_color = self._get_background_color(background)
                
                # 核心处理流程
                np_face, np_mask = self._prepare_inputs(face_image, face_mask)
                
                # 检测人脸中心点，用于旋转和缩放时保持中心不变
                with Timer("人脸分析"):
                    face_info = self._analyze_face(np_face, np_mask)
                    face_center = self._detect_face_center(np_face, np_mask)
                
                # 获取人物比例参数
                params = self._get_person_params(mode, face_info)
                
                # 移除肖像模式下的自动调整face_size限制
                # 让用户可以完全自由地控制face_size参数
                
                # 智能缩放
                with Timer("智能缩放"):
                    scaled_face, scaled_mask, face_center_scaled = self._smart_scaling(
                        np_face, np_mask, target_h, params["head_ratio"], face_size, face_center, face_info
                    )
                
                # 应用旋转
                if angle != 0:
                    with Timer("图像旋转"):
                        scaled_face, scaled_mask, face_center_scaled = self._rotate_image(
                            scaled_face, scaled_mask, angle, bg_color, face_center_scaled
                        )
                
                # 精确定位
                with Timer("精确定位"):
                    final_face, final_mask, final_face_center = self._apply_positioning(
                        scaled_face, scaled_mask, target_w, target_h, bg_color, 
                        move_x, move_y, face_center_scaled, params["eye_position_ratio"]
                    )
                
                # 检查人物是否脱离画布
                self._check_boundary(final_mask, "人物")
                
                # 转换为输出格式
                return self._wrap_outputs(final_face, final_mask)
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建空白图像作为输出
            blank = np.full_like(face_image[0].cpu().numpy(), 0.0)
            blank_tensor = torch.from_numpy(blank).unsqueeze(0)
            return (blank_tensor, face_mask)

    # 工具方法 --------------------------------------------------
    def _calculate_dimensions(self, mode, custom_width, custom_height) -> Tuple[int, int]:
        """智能尺寸计算系统"""
        mode_params = {
            "portrait": (768, 1024),    # 肖像模式 - 3:4比例更适合肖像照
            "half_body": (768, 1024),   # 半身模式
            "full_body": (768, 1024),   # 全身模式
            "custom": (custom_width, custom_height)  # 自定义模式
        }
        
        return mode_params[mode]
    
    def _analyze_face(self, image, mask) -> Dict[str, Any]:
        """分析人脸特征，包括性别估计、人脸大小等"""
        # 计算图像哈希值用于缓存
        image_hash = hash(image.tobytes())
        
        # 检查缓存
        if FaceAutoFitSingle._last_image_hash == image_hash and FaceAutoFitSingle._last_face_info is not None:
            logger.info("使用缓存的人脸分析结果")
            return FaceAutoFitSingle._last_face_info.copy()
        
        face_info = {
            "gender": "unknown",  # 性别: "male", "female", "unknown"
            "face_height": None,  # 人脸高度
            "face_width": None,   # 人脸宽度
            "face_rect": None,    # 人脸矩形 (x, y, w, h)
            "eyes_center": None,  # 眼睛中心点 (x, y)
            "confidence": 0.0     # 检测置信度
        }
        
        # 根据选择的检测方法进行人脸分析
        detection_methods = []
        
        if self.detection_method == "auto":
            detection_methods = ["dlib", "opencv", "mask", "mediapipe"]  # 调整顺序，优先使用dlib
        elif self.detection_method == "none":
            # 仅使用遮罩分析
            detection_methods = ["mask"]
        else:
            detection_methods = [self.detection_method, "mask"]  # 总是添加遮罩作为备选
        
        # 尝试使用选定的方法进行人脸检测
        for method in detection_methods:
            if method == "dlib" and face_info["confidence"] < 0.8:
                self._analyze_face_dlib(image, face_info)
                
                # 如果dlib检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.8:
                    break
                    
            elif method == "opencv" and face_info["confidence"] < 0.7:
                self._analyze_face_opencv(image, face_info)
                
                # 如果OpenCV检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.7:
                    break
                    
            elif method == "mediapipe" and face_info["confidence"] < 0.9:
                self._analyze_face_mediapipe(image, face_info)
                
                # 如果MediaPipe检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.9:
                    break
                    
            elif method == "mask" and face_info["confidence"] < 0.5:
                self._analyze_face_mask(image, mask, face_info)
        
        # 如果仍然没有检测到眼睛中心，使用图像中心点上移1/4处
        if face_info["eyes_center"] is None:
            h, w = image.shape[:2]
            center_x = w // 2
            eyes_y = h // 4
            face_info["eyes_center"] = (center_x, eyes_y)
        
        # 更新缓存
        FaceAutoFitSingle._last_image_hash = image_hash
        FaceAutoFitSingle._last_face_info = face_info.copy()
        
        return face_info
    
    def _analyze_face_opencv(self, image, face_info):
        """使用OpenCV进行人脸分析"""
        try:
            # 延迟加载人脸检测器
            if FaceAutoFitSingle._face_cascade is None:
                FaceAutoFitSingle._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = FaceAutoFitSingle._face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # 获取最大的人脸
                max_area = 0
                max_face = None
                for (x, y, w, h) in faces:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_face = (x, y, w, h)
                
                x, y, w, h = max_face
                face_info["face_width"] = w
                face_info["face_height"] = h
                face_info["face_rect"] = (x, y, w, h)
                face_info["confidence"] = 0.7
                
                # 估计眼睛位置 - OpenCV只能粗略估计
                eyes_x = x + w // 2
                eyes_y = y + int(h * 0.4)  # 眼睛通常在人脸上部40%处
                face_info["eyes_center"] = (eyes_x, eyes_y)
                
                # 简单的性别估计 - 基于脸部宽高比
                face_ratio = w / h if h > 0 else 0
                if face_ratio > 0.78:  # 男性脸部通常更宽
                    face_info["gender"] = "male"
                else:
                    face_info["gender"] = "female"
                
                return True
        except Exception as e:
            logger.warning(f"OpenCV人脸分析失败: {str(e)}")
        
        return False
    
    def _analyze_face_mediapipe(self, image, face_info):
        """使用MediaPipe进行人脸分析"""
        try:
            # 延迟导入和加载MediaPipe
            import mediapipe as mp
            
            if FaceAutoFitSingle._mediapipe_face_mesh is None:
                mp_face_mesh = mp.solutions.face_mesh
                FaceAutoFitSingle._mediapipe_face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5)
            
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb_image.shape[:2]
            
            # 处理图像
            mp_results = FaceAutoFitSingle._mediapipe_face_mesh.process(rgb_image)
            
            if mp_results.multi_face_landmarks:
                landmarks = mp_results.multi_face_landmarks[0].landmark
                
                # 计算人脸边界框
                x_min = min(landmark.x for landmark in landmarks) * w
                y_min = min(landmark.y for landmark in landmarks) * h
                x_max = max(landmark.x for landmark in landmarks) * w
                y_max = max(landmark.y for landmark in landmarks) * h
                
                face_width = x_max - x_min
                face_height = y_max - y_min
                
                # 更新人脸信息，因为MediaPipe通常比OpenCV更准确
                face_info["face_width"] = face_width
                face_info["face_height"] = face_height
                face_info["face_rect"] = (int(x_min), int(y_min), int(face_width), int(face_height))
                face_info["confidence"] = 0.9
                
                # MediaPipe中眼睛关键点的索引
                LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 左眼轮廓点
                RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 右眼轮廓点
                
                # 计算左眼中心
                left_eye_x = sum(landmarks[idx].x for idx in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES)
                left_eye_y = sum(landmarks[idx].y for idx in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES)
                
                # 计算右眼中心
                right_eye_x = sum(landmarks[idx].x for idx in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES)
                right_eye_y = sum(landmarks[idx].y for idx in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES)
                
                # 计算两眼中心
                eyes_center_x = int((left_eye_x + right_eye_x) * w / 2)
                eyes_center_y = int((left_eye_y + right_eye_y) * h / 2)
                
                face_info["eyes_center"] = (eyes_center_x, eyes_center_y)
                
                # 使用眉间距离估计性别
                # 眉毛关键点
                LEFT_EYEBROW = 336   # 左眉中点
                RIGHT_EYEBROW = 107  # 右眉中点
                
                # 计算眉间距离与脸宽的比例
                brow_distance = abs(landmarks[LEFT_EYEBROW].x - landmarks[RIGHT_EYEBROW].x) * w
                brow_face_ratio = brow_distance / face_width if face_width > 0 else 0
                
                # 简化性别判断
                if brow_face_ratio < 0.15:  # 眉间距离较小，可能是男性
                    face_info["gender"] = "male"
                else:
                    face_info["gender"] = "female"
                
                return True
        except Exception as e:
            logger.warning(f"MediaPipe人脸分析失败: {str(e)}")
        
        return False
    
    def _analyze_face_dlib(self, image, face_info):
        """使用dlib进行人脸分析"""
        try:
            import dlib
            
            # 延迟加载dlib模型
            if FaceAutoFitSingle._dlib_detector is None:
                FaceAutoFitSingle._dlib_detector = dlib.get_frontal_face_detector()
                
                # 获取模型文件路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                predictor_path = os.path.join(current_dir, "models", "shape_predictor_68_face_landmarks.dat")
                
                if os.path.exists(predictor_path):
                    FaceAutoFitSingle._dlib_predictor = dlib.shape_predictor(predictor_path)
                else:
                    logger.warning(f"dlib模型文件不存在: {predictor_path}")
                    return False
            
            # 性能优化：缩小图像进行检测
            h, w = image.shape[:2]
            scale = 1.0
            
            # 如果图像较大，先缩小进行检测
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small_img = image
            
            # 转换为灰度图
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸 - 使用更快的参数
            # 第二个参数是上采样次数，0表示不上采样，速度更快但可能检测不到小人脸
            faces = FaceAutoFitSingle._dlib_detector(gray, 0)
            
            if len(faces) > 0:
                # 获取最大的人脸
                max_area = 0
                max_face = None
                for face in faces:
                    area = (face.right() - face.left()) * (face.bottom() - face.top())
                    if area > max_area:
                        max_area = area
                        max_face = face
                
                # 获取人脸关键点
                shape = FaceAutoFitSingle._dlib_predictor(gray, max_face)
                
                # 计算人脸宽度和高度
                if scale != 1.0:
                    face_width = int((max_face.right() - max_face.left()) / scale)
                    face_height = int((max_face.bottom() - max_face.top()) / scale)
                    face_rect = (
                        int(max_face.left() / scale), 
                        int(max_face.top() / scale), 
                        face_width, 
                        face_height
                    )
                else:
                    face_width = max_face.right() - max_face.left()
                    face_height = max_face.bottom() - max_face.top()
                    face_rect = (max_face.left(), max_face.top(), face_width, face_height)
                
                face_info["face_width"] = face_width
                face_info["face_height"] = face_height
                face_info["face_rect"] = face_rect
                face_info["confidence"] = 0.8
                
                # 计算眼睛中心点
                left_eye_x = sum(shape.part(i).x for i in range(36, 42)) / 6
                left_eye_y = sum(shape.part(i).y for i in range(36, 42)) / 6
                right_eye_x = sum(shape.part(i).x for i in range(42, 48)) / 6
                right_eye_y = sum(shape.part(i).y for i in range(42, 48)) / 6
                
                if scale != 1.0:
                    eyes_center_x = int((left_eye_x + right_eye_x) / (2 * scale))
                    eyes_center_y = int((left_eye_y + right_eye_y) / (2 * scale))
                else:
                    eyes_center_x = int((left_eye_x + right_eye_x) / 2)
                    eyes_center_y = int((left_eye_y + right_eye_y) / 2)
                
                face_info["eyes_center"] = (eyes_center_x, eyes_center_y)
                
                # 使用眉间距离估计性别
                if scale != 1.0:
                    brow_distance = abs(int(shape.part(21).x / scale) - int(shape.part(22).x / scale))
                else:
                    brow_distance = abs(shape.part(21).x - shape.part(22).x)
                
                brow_face_ratio = brow_distance / face_width if face_width > 0 else 0
                
                # 简化性别判断
                if brow_face_ratio < 0.15:  # 眉间距离较小，可能是男性
                    face_info["gender"] = "male"
                else:
                    face_info["gender"] = "female"
                
                return True
        except Exception as e:
            logger.warning(f"dlib人脸分析失败: {str(e)}")
        
        return False
    
    def _analyze_face_mask(self, image, mask, face_info):
        """使用遮罩分析人脸"""
        try:
            if np.max(mask) > 0:
                # 计算遮罩的非零区域
                non_zero_points = np.argwhere(mask > 0)
                if len(non_zero_points) > 0:
                    y_min, x_min = non_zero_points.min(axis=0)
                    y_max, x_max = non_zero_points.max(axis=0)
                    
                    mask_height = y_max - y_min
                    mask_width = x_max - x_min
                    
                    # 估计人脸高度为遮罩高度的一部分
                    face_info["face_height"] = mask_height * 0.4  # 假设人脸占遮罩高度的40%
                    face_info["face_width"] = mask_width * 0.4    # 假设人脸占遮罩宽度的40%
                    face_info["face_rect"] = (x_min, y_min, mask_width, mask_height)
                    face_info["confidence"] = 0.5
                    
                    # 估计眼睛位置在遮罩上部的1/4处
                    mask_center_x = (x_min + x_max) // 2
                    eyes_y = int(y_min + mask_height * 0.25)
                    face_info["eyes_center"] = (mask_center_x, eyes_y)
                    
                    return True
        except Exception as e:
            logger.warning(f"遮罩分析失败: {str(e)}")
        
        return False

    def _get_person_params(self, mode, face_info) -> Dict[str, float]:
        """获取人物比例参数 - 基于人脸分析而非性别假设"""
        # 默认参数 - 适用于未知性别
        params = {
            "head_ratio": 0.15,           # 头部占画布高度的比例
            "eye_position_ratio": 0.30    # 眼睛位于画布高度的比例
        }
        
        # 根据模式调整基础参数
        if mode == "portrait":
            # 肖像模式 - 优化构图
            params["head_ratio"] = 0.25    # 增大头部比例，从0.15调整为0.25
            params["eye_position_ratio"] = 0.33  # 将眼睛位置调整到三分线位置（约1/3处）
        elif mode == "half_body":
            # 半身模式 - 更精确地调整眼睛位置和人脸比例
            params["head_ratio"] = 0.15    # 调整头部比例，更小一些以留出更多身体空间
            params["eye_position_ratio"] = 0.22  # 将眼睛位置调高，约为画面高度的22%处
        elif mode == "full_body":
            # 全身模式
            params["head_ratio"] = 0.10
            params["eye_position_ratio"] = 0.15
        
        # 根据性别微调参数
        if face_info["gender"] == "male":
            # 男性通常头部略大，眼睛位置略低
            params["head_ratio"] *= 1.05
            params["eye_position_ratio"] *= 0.95
        elif face_info["gender"] == "female":
            # 女性通常头部略小，眼睛位置略高
            params["head_ratio"] *= 0.95
            params["eye_position_ratio"] *= 1.05
        
        logger.info(f"人物参数: 性别={face_info['gender']}, 头部比例={params['head_ratio']:.3f}, 眼睛位置={params['eye_position_ratio']:.3f}")
        
        return params
    
    def _get_background_color(self, bg_type) -> Tuple[int, int, int]:
        """背景颜色管理系统"""
        return (255, 255, 255) if bg_type == "white" else (192, 192, 192)
    
    def _prepare_inputs(self, image_tensor, mask_tensor) -> Tuple[np.ndarray, np.ndarray]:
        """输入预处理"""
        image = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)[:, :, :3]
        mask = (mask_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        return image, mask
    
    def _detect_face_center(self, image, mask) -> Tuple[int, int]:
        """检测人脸中心点 - 优化版"""
        # 检查是否有缓存的人脸信息
        if FaceAutoFitSingle._last_face_info is not None and FaceAutoFitSingle._last_face_info["eyes_center"] is not None:
            return FaceAutoFitSingle._last_face_info["eyes_center"]
        
        # 如果没有缓存，使用快速检测
        try:
            # 使用mask的质心作为快速估计
            if np.max(mask) > 0:
                # 计算mask的质心
                M = cv2.moments(mask)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        except Exception as e:
            logger.warning(f"质心计算失败: {str(e)}")
        
        # 如果都失败了，使用图像中心
        h, w = image.shape[:2]
        return (w//2, h//2)
    
    def _rotate_image(self, image, mask, angle, bg_color, center) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """以指定中心点旋转图像，保持面部中心不变"""
        if angle == 0:
            return image, mask, center
            
        h, w = image.shape[:2]
        cx, cy = center
        
        # 计算旋转后需要的画布大小，确保旋转后的图像完全显示
        angle_rad = abs(angle) * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 计算旋转后的新宽度和高度
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # 创建一个更大的画布
        padded_img = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)
        padded_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # 计算原图在新画布中的位置，使得旋转中心保持在相同的相对位置
        pad_x = (new_w - w) // 2
        pad_y = (new_h - h) // 2
        
        # 将原图放在新画布中
        padded_img[pad_y:pad_y+h, pad_x:pad_x+w] = image
        padded_mask[pad_y:pad_y+h, pad_x:pad_x+w] = mask
        
        # 更新旋转中心点在新画布中的坐标
        new_cx = cx + pad_x
        new_cy = cy + pad_y
        
        # 创建旋转矩阵，以新的中心点为旋转中心
        matrix = cv2.getRotationMatrix2D((new_cx, new_cy), -angle, 1)
        
        # 旋转时填充背景色
        rotated_img = cv2.warpAffine(padded_img, matrix, (new_w, new_h), 
                                   flags=cv2.INTER_LANCZOS4,
                                   borderValue=bg_color)
        rotated_mask = cv2.warpAffine(padded_mask, matrix, (new_w, new_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderValue=0)
        
        # 旋转后的中心点坐标保持不变，因为我们是以该点为中心进行的旋转
        rotated_center = (new_cx, new_cy)
        
        return rotated_img, rotated_mask, rotated_center
    
    def _smart_scaling(self, image, mask, target_h, head_ratio, face_size, face_center, face_info) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """智能缩放算法 - 根据人脸比例调整，保持人脸中心不变"""
        # 优先使用检测到的人脸区域进行缩放
        if face_info["face_height"] is not None:
            try:
                # 使用人脸高度计算缩放比例
                face_height = face_info["face_height"]
                target_face_height = target_h * head_ratio
                scale_factor = (target_face_height / face_height) * face_size
                
                # 强制限制缩放比例在合理范围内
                min_scale = 0.1
                max_scale = 2.0
                scale_factor = max(min(scale_factor, max_scale), min_scale)
                
                # 应用缩放
                new_h = int(image.shape[0] * scale_factor)
                new_w = int(image.shape[1] * scale_factor)
                
                # 确保尺寸至少为1像素
                new_h = max(1, new_h)
                new_w = max(1, new_w)
                
                # 使用高质量插值方法进行缩放
                scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # 计算缩放后的人脸中心点
                if face_info["eyes_center"] is not None:
                    # 使用眼睛中心点作为参考
                    eyes_x, eyes_y = face_info["eyes_center"]
                    scaled_eyes_x = int(eyes_x * scale_factor)
                    scaled_eyes_y = int(eyes_y * scale_factor)
                    return scaled_img, scaled_mask, (scaled_eyes_x, scaled_eyes_y)
                else:
                    # 使用人脸中心点作为参考
                    cx, cy = face_center
                    scaled_cx = int(cx * scale_factor)
                    scaled_cy = int(cy * scale_factor)
                    return scaled_img, scaled_mask, (scaled_cx, scaled_cy)
            except Exception as e:
                logger.warning(f"基于人脸的缩放失败: {str(e)}")
        
        # 如果人脸检测失败，使用整个遮罩进行缩放
        try:
            if np.max(mask) > 0:
                # 计算遮罩的边界框
                non_zero_points = np.argwhere(mask > 0)
                y_min, x_min = non_zero_points.min(axis=0)
                y_max, x_max = non_zero_points.max(axis=0)
                
                mask_height = y_max - y_min
                
                # 使用遮罩高度计算缩放比例
                # 假设遮罩高度的40%是人脸高度
                estimated_face_height = mask_height * 0.4
                target_face_height = target_h * head_ratio
                scale_factor = (target_face_height / estimated_face_height) * face_size
                
                # 强制限制缩放比例在合理范围内
                min_scale = 0.1
                max_scale = 2.0
                scale_factor = max(min(scale_factor, max_scale), min_scale)
                
                # 应用缩放
                new_h = int(image.shape[0] * scale_factor)
                new_w = int(image.shape[1] * scale_factor)
                
                # 确保尺寸至少为1像素
                new_h = max(1, new_h)
                new_w = max(1, new_w)
                
                # 使用高质量插值方法进行缩放
                scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # 计算缩放后的中心点
                cx, cy = face_center
                scaled_cx = int(cx * scale_factor)
                scaled_cy = int(cy * scale_factor)
                
                return scaled_img, scaled_mask, (scaled_cx, scaled_cy)
        except Exception as e:
            logger.warning(f"基于遮罩的缩放失败: {str(e)}")
        
        # 如果所有方法都失败，使用默认缩放
        try:
            # 默认缩放 - 使整个图像高度为目标高度的70%
            scale_factor = (target_h * 0.7) / image.shape[0] * face_size
            
            # 强制限制缩放比例在合理范围内
            min_scale = 0.1
            max_scale = 2.0
            scale_factor = max(min(scale_factor, max_scale), min_scale)
            
            # 应用缩放
            new_h = int(image.shape[0] * scale_factor)
            new_w = int(image.shape[1] * scale_factor)
            
            # 确保尺寸至少为1像素
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            
            # 使用高质量插值方法进行缩放
            scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 计算缩放后的中心点
            h, w = image.shape[:2]
            cx, cy = w//2, h//2
            scaled_cx = int(cx * scale_factor)
            scaled_cy = int(cy * scale_factor)
            
            return scaled_img, scaled_mask, (scaled_cx, scaled_cy)
        except Exception as e:
            logger.error(f"默认缩放失败: {str(e)}")
            return image, mask, face_center
    
    def _apply_positioning(self, image, mask, target_w, target_h, bg_color, move_x, move_y, face_center, eye_position_ratio) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """精确定位系统 - 根据眼睛位置和用户调整"""
        # 创建画布
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        mask_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # 计算水平位置 - 居中
        A_x = int(target_w * 0.5)  # 眼睛在画布宽度的中心
            
        # 计算垂直位置 - 眼睛在画布高度的eye_position_ratio处
        A_y = int(target_h * eye_position_ratio)
        
        # 确保face_center是有效的
        if face_center is None:
            # 如果face_center无效，使用图像中心
            h, w = image.shape[:2]
            face_center = (w//2, h//2)
            logger.warning(f"使用图像中心作为人脸中心点的回退方案")
        
        # 获取人脸中心点
        B_x, B_y = face_center
        
        # 计算基础偏移量 - 将B点移动到A点
        base_offset_x = A_x - B_x
        base_offset_y = A_y - B_y
        
        # 应用用户的额外位置调整
        user_offset_x = int(move_x * target_w / 200)  # 将-100到100映射到-50%到50%的画布宽度
        user_offset_y = int(-move_y * target_h / 200)  # 注意负号，使正值向上移动
        
        # 计算最终偏移量
        final_offset_x = base_offset_x + user_offset_x
        final_offset_y = base_offset_y + user_offset_y
        
        # 计算粘贴区域
        x_start = final_offset_x
        y_start = final_offset_y
        x_end = x_start + image.shape[1]
        y_end = y_start + image.shape[0]
        
        # 计算源图像的裁剪区域
        src_x_start = 0
        src_y_start = 0
        
        if x_start < 0:
            src_x_start = -x_start
            x_start = 0
        if y_start < 0:
            src_y_start = -y_start
            y_start = 0
            
        if x_end > target_w:
            src_x_end = image.shape[1] - (x_end - target_w)
            x_end = target_w
        else:
            src_x_end = image.shape[1]
            
        if y_end > target_h:
            src_y_end = image.shape[0] - (y_end - target_h)
            y_end = target_h
        else:
            src_y_end = image.shape[0]
        
        # 仅粘贴有效区域
        if (x_end > x_start and y_end > y_start and 
            src_x_end > src_x_start and src_y_end > src_y_start):
            
            img_region = image[src_y_start:src_y_end, src_x_start:src_x_end]
            mask_region = mask[src_y_start:src_y_end, src_x_start:src_x_end]
            
            if img_region.shape[0] > 0 and img_region.shape[1] > 0:
                # 使用mask进行混合
                mask_3d = np.expand_dims(mask_region, axis=2) / 255.0
                canvas_region = canvas[y_start:y_end, x_start:x_end]
                blended = img_region * mask_3d + canvas_region * (1 - mask_3d)
                
                canvas[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
                mask_canvas[y_start:y_end, x_start:x_end] = mask_region
        
        # 计算最终人脸中心在画布上的位置
        final_face_center_x = A_x + user_offset_x
        final_face_center_y = A_y + user_offset_y
        
        return canvas, mask_canvas, (final_face_center_x, final_face_center_y)
    
    def _check_boundary(self, mask, name):
        """检查人物是否脱离画布边界"""
        if np.max(mask) == 0:
            return
            
        h, w = mask.shape
        
        # 检查上下左右边界是否有非零像素
        top_row = mask[0, :]
        bottom_row = mask[h-1, :]
        left_col = mask[:, 0]
        right_col = mask[:, w-1]
        
        warnings = []
        if np.max(top_row) > 0:
            warnings.append("上")
        if np.max(bottom_row) > 0:
            warnings.append("下")
        if np.max(left_col) > 0:
            warnings.append("左")
        if np.max(right_col) > 0:
            warnings.append("右")
            
        if warnings:
            logger.warning(f"{name}在画布{', '.join(warnings)}边界处被裁剪，请调整位置或缩小尺寸")
    
    def _wrap_outputs(self, image, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """输出格式转换"""
        return (
            torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0),
            torch.from_numpy(mask.astype(np.float32)/255.0).unsqueeze(0)
        )

# 将这两行移到类外部
NODE_CLASS_MAPPINGS = {"FaceAutoFitSingle": FaceAutoFitSingle}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAutoFitSingle": "Face AutoFit Single @ CHAOS"}