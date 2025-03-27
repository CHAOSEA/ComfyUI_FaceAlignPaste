import cv2
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .logging_utils import logger

class FaceDetector:
    # 类变量用于缓存模型
    _face_cascade = None
    _mediapipe_face_mesh = None
    _dlib_detector = None
    _dlib_predictor = None
    _face_analyzer = None  # InsightFace模型
    
    # 缓存变量
    _last_image_hash = None
    _last_face_info = None
    
    @staticmethod
    def analyze_face(image, mask, detection_method="insightface"):
        """分析人脸，返回人脸信息"""
        # 计算图像哈希值，用于缓存
        image_hash = hash(image.tobytes())
        
        # 检查缓存
        if FaceDetector._last_image_hash == image_hash and FaceDetector._last_face_info is not None:
            return FaceDetector._last_face_info.copy()
        
        # 初始化人脸信息
        face_info = {
            "face_width": None,      # 人脸宽度
            "face_height": None,     # 人脸高度
            "face_rect": None,       # 人脸矩形 (x, y, w, h)
            "eyes_center": None,     # 眼睛中心点 (x, y)
            "gender": "unknown",     # 性别
            "gender_confidence": 0.0, # 性别置信度
            "confidence": 0.0        # 检测置信度
        }
        
        # 根据选择的检测方法进行人脸分析
        detection_methods = []
        
        if detection_method == "auto":
            detection_methods = ["insightface", "dlib", "opencv", "mediapipe", "mask"]
        elif detection_method == "none":
            # 仅使用遮罩分析
            detection_methods = ["mask"]
        else:
            detection_methods = [detection_method, "mask"]  # 总是添加遮罩作为备选
        
        # 尝试使用选定的方法进行人脸检测
        for method in detection_methods:
            if method == "insightface" and face_info["confidence"] < 0.9:
                FaceDetector._analyze_face_insightface(image, face_info)
                
                # 如果InsightFace检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.9:
                    break
                    
            elif method == "dlib" and face_info["confidence"] < 0.8:
                FaceDetector._analyze_face_dlib(image, face_info)
                
                # 如果dlib检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.8:
                    break
                    
            elif method == "opencv" and face_info["confidence"] < 0.7:
                FaceDetector._analyze_face_opencv(image, face_info)
                
                # 如果OpenCV检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.7:
                    break
                    
            elif method == "mediapipe" and face_info["confidence"] < 0.9:
                FaceDetector._analyze_face_mediapipe(image, face_info)
                
                # 如果MediaPipe检测成功且置信度足够高，就不再尝试其他方法
                if face_info["confidence"] >= 0.9:
                    break
                    
            elif method == "mask" and face_info["confidence"] < 0.5:
                FaceDetector._analyze_face_mask(image, mask, face_info)
        
        # 如果仍然没有检测到眼睛中心，使用图像中心点上移1/4处
        if face_info["eyes_center"] is None:
            h, w = image.shape[:2]
            center_x = w // 2
            eyes_y = h // 4
            face_info["eyes_center"] = (center_x, eyes_y)
        
        # 更新缓存
        FaceDetector._last_image_hash = image_hash
        FaceDetector._last_face_info = face_info.copy()
        
        return face_info
    
    @staticmethod
    def _analyze_face_insightface(image, face_info):
        """使用InsightFace进行人脸分析"""
        try:
            # 延迟导入InsightFace
            from insightface.app import FaceAnalysis
            
            # 延迟加载InsightFace模型
            if FaceDetector._face_analyzer is None:
                FaceDetector._face_analyzer = FaceDetector._load_insightface()
                if FaceDetector._face_analyzer is None:
                    logger.warning("InsightFace模型加载失败，跳过InsightFace分析")
                    return False
            
            # 性能优化：缩小图像进行检测
            h, w = image.shape[:2]
            scale = 1.0
            
            # 如果图像较大，先缩小进行检测（与dlib类似的优化）
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small_img = image
            
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            
            # 分析人脸
            faces = FaceDetector._face_analyzer.get(rgb_image)
            
            if len(faces) > 0:
                # 获取最大的人脸
                max_area = 0
                max_face = None
                for face in faces:
                    bbox = face.bbox.astype(int)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_face = face
                
                if max_face is not None:
                    # 获取人脸边界框
                    bbox = max_face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    
                    # 如果进行了缩放，恢复原始坐标
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    
                    w, h = x2 - x, y2 - y
                    
                    # 更新人脸信息
                    face_info["face_width"] = w
                    face_info["face_height"] = h
                    face_info["face_rect"] = (x, y, w, h)
                    face_info["confidence"] = float(max_face.det_score)
                    
                    # 获取关键点
                    landmarks = max_face.landmark_2d_106
                    if landmarks is not None and len(landmarks) > 0:
                        # 计算眼睛中心点
                        left_eye = landmarks[96]  # 左眼中心
                        right_eye = landmarks[97]  # 右眼中心
                        
                        # 如果进行了缩放，恢复原始坐标
                        if scale != 1.0:
                            eyes_center_x = int((left_eye[0] / scale + right_eye[0] / scale) / 2)
                            eyes_center_y = int((left_eye[1] / scale + right_eye[1] / scale) / 2)
                        else:
                            eyes_center_x = int((left_eye[0] + right_eye[0]) / 2)
                            eyes_center_y = int((left_eye[1] + right_eye[1]) / 2)
                        
                        face_info["eyes_center"] = (eyes_center_x, eyes_center_y)
                    
                    # 获取性别
                    if hasattr(max_face, 'gender') and max_face.gender is not None:
                        gender_idx = int(max_face.gender)
                        gender = "male" if gender_idx == 1 else "female"
                        face_info["gender"] = gender
                        face_info["gender_confidence"] = 0.9  # InsightFace没有提供性别置信度
                        logger.info(f"InsightFace检测性别: {gender}")
                    
                    return True
        except Exception as e:
            logger.warning(f"InsightFace人脸分析失败: {str(e)}")
        
        return False
    
    @staticmethod
    def _analyze_face_opencv(image, face_info):
        """使用OpenCV进行人脸分析"""
        try:
            # 延迟加载人脸检测器
            if FaceDetector._face_cascade is None:
                FaceDetector._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = FaceDetector._face_cascade.detectMultiScale(gray, 1.1, 4)
            
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
    
    @staticmethod
    def _analyze_face_mediapipe(image, face_info):
        """使用MediaPipe进行人脸分析"""
        try:
            # 延迟导入和加载MediaPipe
            import mediapipe as mp
            
            if FaceDetector._mediapipe_face_mesh is None:
                mp_face_mesh = mp.solutions.face_mesh
                FaceDetector._mediapipe_face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5)
            
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb_image.shape[:2]
            
            # 处理图像
            mp_results = FaceDetector._mediapipe_face_mesh.process(rgb_image)
            
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
                left_eye_x = sum(landmarks[idx].x for idx in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES) * w
                left_eye_y = sum(landmarks[idx].y for idx in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES) * h
                
                # 计算右眼中心
                right_eye_x = sum(landmarks[idx].x for idx in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES) * w
                right_eye_y = sum(landmarks[idx].y for idx in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES) * h
                
                # 计算两眼中心
                eyes_center_x = int((left_eye_x + right_eye_x) / 2)
                eyes_center_y = int((left_eye_y + right_eye_y) / 2)
                
                face_info["eyes_center"] = (eyes_center_x, eyes_center_y)
                
                # 性别估计 - 基于多个特征点
                # 计算下巴宽度与脸部高度的比例
                jaw_width = abs(landmarks[152].x - landmarks[377].x) * w
                face_height = abs(landmarks[10].y - landmarks[152].y) * h
                jaw_face_ratio = jaw_width / face_height if face_height > 0 else 0
                
                # 计算眉间距离与脸部宽度的比例
                brow_distance = abs(landmarks[107].x - landmarks[336].x) * w
                brow_face_ratio = brow_distance / face_width if face_width > 0 else 0
                
                # 基于多个特征估计性别
                if jaw_face_ratio > 0.85 and brow_face_ratio < 0.4:
                    face_info["gender"] = "male"
                elif jaw_face_ratio < 0.8 and brow_face_ratio > 0.35:
                    face_info["gender"] = "female"
                
                return True
        except Exception as e:
            logger.warning(f"MediaPipe人脸分析失败: {str(e)}")
        
        return False
    
    @staticmethod
    def _analyze_face_dlib(image, face_info):
        """使用dlib进行人脸分析"""
        try:
            # 延迟导入dlib
            import dlib
            
            # 获取模型文件路径
            predictor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shape_predictor_68_face_landmarks.dat")
            
            if not os.path.exists(predictor_path):
                logger.warning(f"dlib模型文件不存在: {predictor_path}")
                return False
                
            # 延迟加载dlib模型
            if FaceDetector._dlib_detector is None:
                FaceDetector._dlib_detector = dlib.get_frontal_face_detector()
                
            if FaceDetector._dlib_predictor is None:
                FaceDetector._dlib_predictor = dlib.shape_predictor(predictor_path)
            
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
            faces = FaceDetector._dlib_detector(gray, 0)
            
            if len(faces) > 0:
                # 获取人脸关键点
                shape = FaceDetector._dlib_predictor(gray, faces[0])
                
                # 计算人脸边界框
                x_min = min(shape.part(i).x for i in range(shape.num_parts))
                y_min = min(shape.part(i).y for i in range(shape.num_parts))
                x_max = max(shape.part(i).x for i in range(shape.num_parts))
                y_max = max(shape.part(i).y for i in range(shape.num_parts))
                
                # 如果进行了缩放，恢复原始坐标
                if scale != 1.0:
                    x_min = int(x_min / scale)
                    y_min = int(y_min / scale)
                    x_max = int(x_max / scale)
                    y_max = int(y_max / scale)
                
                face_width = x_max - x_min
                face_height = y_max - y_min
                
                # 更新人脸信息
                face_info["face_width"] = face_width
                face_info["face_height"] = face_height
                face_info["face_rect"] = (x_min, y_min, face_width, face_height)
                face_info["confidence"] = 0.8
                
                # 计算眼睛中心点
                left_eye_center_idx = 37  # 左眼中心点索引
                right_eye_center_idx = 44  # 右眼中心点索引
                
                # 如果进行了缩放，恢复原始坐标
                if scale != 1.0:
                    left_eye_x = int(shape.part(left_eye_center_idx).x / scale)
                    left_eye_y = int(shape.part(left_eye_center_idx).y / scale)
                    right_eye_x = int(shape.part(right_eye_center_idx).x / scale)
                    right_eye_y = int(shape.part(right_eye_center_idx).y / scale)
                else:
                    left_eye_x = shape.part(left_eye_center_idx).x
                    left_eye_y = shape.part(left_eye_center_idx).y
                    right_eye_x = shape.part(right_eye_center_idx).x
                    right_eye_y = shape.part(right_eye_center_idx).y
                
                # 计算两眼中心
                eyes_center_x = int((left_eye_x + right_eye_x) / 2)
                eyes_center_y = int((left_eye_y + right_eye_y) / 2)
                
                face_info["eyes_center"] = (eyes_center_x, eyes_center_y)
                
                # 性别估计 - 简化计算
                # 使用眉间距离与脸部宽度的比例作为性别特征
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
    
    @staticmethod
    def _analyze_face_mask(image, mask, face_info):
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


    @staticmethod
    def _load_insightface():
        """加载InsightFace模型"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import os
            
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 获取ComfyUI根目录
            comfyui_root = os.path.dirname(os.path.dirname(current_dir))
            # 设置模型路径为相对路径
            models_dir = os.path.join(comfyui_root, "models", "insightface")
            buffalo_path = os.path.join(models_dir, "models", "buffalo_l")
            
            # 检查模型文件是否存在
            required_files = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]
            models_exist = True
            
            # 确保目录存在
            if not os.path.exists(buffalo_path):
                os.makedirs(buffalo_path, exist_ok=True)
                models_exist = False
            else:
                # 检查所有必需的模型文件
                for file in required_files:
                    if not os.path.exists(os.path.join(buffalo_path, file)):
                        models_exist = False
                        break
            
            if models_exist:
                logger.info(f"InsightFace模型已存在，直接使用本地模型: {buffalo_path}")
            else:
                logger.info(f"未找到完整的InsightFace模型，将下载到: {buffalo_path}")
            
            # 设置InsightFace模型路径
            os.environ['INSIGHTFACE_HOME'] = models_dir
            
            # 创建分析器 - 使用更小的检测尺寸以提高速度
            face_analyzer = FaceAnalysis(
                name="buffalo_l",
                root=models_dir,
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'landmark_2d_106', 'genderage']  # 只加载必要的模块
            )
            
            # 准备模型 - 使用更小的检测尺寸
            face_analyzer.prepare(ctx_id=0, det_size=(320, 320))  # 降低检测尺寸以提高速度
            
            if models_exist:
                logger.info("InsightFace模型加载成功")
            else:
                logger.info(f"InsightFace模型下载并加载成功，保存在: {buffalo_path}")
                
            return face_analyzer
        except Exception as e:
            logger.error(f"InsightFace模型加载失败: {str(e)}")