import numpy as np
import torch
from PIL import Image
import cv2
import math
import os
import logging
import time
from typing import Tuple, Optional, Dict, Any, List
import mediapipe as mp

try:
    import dlib
    has_dlib = True
except ImportError:
    has_dlib = False
    print("建议安装dlib库作为备用检测器: pip install dlib")

try:
    from face_alignment import FaceAlignment, LandmarksType
    has_face_alignment = True
except ImportError:
    has_face_alignment = False
    print("请安装face_alignment库: pip install face-alignment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceAlignSingle")

class PerformanceTimer:
    """简单的性能计时器类"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logger.info(f"[性能] {self.name}: {elapsed:.4f}秒")

class FaceAlignSingle:
    """Face alignment and pasting node for ComfyUI (Single Face Version)"""
    
    def __init__(self):
        with PerformanceTimer("初始化人脸检测模型"):
            # 使用MediaPipe Face Mesh作为主要检测器
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,  # 只检测一个人脸
                refine_landmarks=True,  # 使用精细特征点以提高准确性
                min_detection_confidence=0.5
            )
            
            # 初始化dlib检测器作为备用
            if has_dlib:
                # 获取模型文件路径 - 修改为使用models子目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(current_dir, "models")
                
                # 如果models目录不存在，创建它
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                    logger.info(f"创建models目录: {models_dir}")
                
                shape_predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
                
                # 检查模型文件是否存在
                if not os.path.exists(shape_predictor_path):
                    logger.warning(f"dlib模型文件不存在: {shape_predictor_path}")
                    logger.warning("请从http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2下载并解压到models目录")
                    self.dlib_detector = None
                    self.dlib_predictor = None
                else:
                    logger.info(f"加载dlib模型: {shape_predictor_path}")
                    self.dlib_detector = dlib.get_frontal_face_detector()
                    self.dlib_predictor = dlib.shape_predictor(shape_predictor_path)
                    logger.info("dlib模型加载成功")
            else:
                self.dlib_detector = None
                self.dlib_predictor = None
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 缓存机制
        self.landmarks_cache = {}
        self.last_processed_images = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "source": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05
                }),
                "move_x": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "move_y": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "raw_mask")
    FUNCTION = "align_and_paste_face"
    CATEGORY = "image/face"

    def get_face_landmarks(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """使用MediaPipe和dlib获取人脸特征点"""
        # 创建图像哈希作为缓存键
        h, w = image.shape[:2]
        # 只使用中心区域计算哈希，提高速度
        center_crop = image[h//2-50:h//2+50, w//2-50:w//2+50, :] if h > 100 and w > 100 else image
        downsampled = center_crop[::8, ::8, :]
        image_hash = hash(downsampled.tobytes())
        
        # 检查缓存
        if image_hash in self.landmarks_cache:
            logger.info("使用缓存的人脸特征点")
            return self.landmarks_cache[image_hash]
        
        with PerformanceTimer("人脸特征点检测"):
            # 准备RGB图像用于MediaPipe
            if image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # 1. 首先使用MediaPipe检测
            mediapipe_landmarks = None
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # 转换MediaPipe特征点格式
                mediapipe_landmarks = []
                for face_landmarks in results.multi_face_landmarks:
                    face_points = []
                    for landmark in face_landmarks.landmark:
                        # 转换为图像坐标
                        x = landmark.x * image.shape[1]
                        y = landmark.y * image.shape[0]
                        face_points.append([x, y])
                    
                    mediapipe_landmarks.append(np.array(face_points))
                logger.info("MediaPipe成功检测到人脸")
            else:
                logger.warning("MediaPipe未检测到人脸，尝试使用dlib")
            
            # 2. 如果MediaPipe失败或者我们有dlib，也用dlib检测
            dlib_landmarks = None
            if self.dlib_detector is not None and (mediapipe_landmarks is None or len(mediapipe_landmarks) == 0):
                # 转换为灰度图像用于dlib
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 检测人脸
                dlib_faces = self.dlib_detector(gray)
                
                if len(dlib_faces) > 0:
                    # 获取第一个人脸的特征点
                    dlib_landmarks = []
                    for face in dlib_faces:
                        shape = self.dlib_predictor(gray, face)
                        points = []
                        for i in range(68):  # dlib有68个特征点
                            x = shape.part(i).x
                            y = shape.part(i).y
                            points.append([x, y])
                        dlib_landmarks.append(np.array(points))
                    logger.info(f"dlib成功检测到{len(dlib_landmarks)}个人脸")
            
            # 3. 决定使用哪个检测结果
            final_landmarks = None
            
            if mediapipe_landmarks and len(mediapipe_landmarks) > 0:
                if dlib_landmarks and len(dlib_landmarks) > 0:
                    # 两者都检测到了，检查一致性
                    # 比较MediaPipe和dlib的眼睛中心位置
                    mp_left_eye = np.mean(mediapipe_landmarks[0][[33, 133, 157, 158, 159, 160, 161, 173, 246]], axis=0)
                    mp_right_eye = np.mean(mediapipe_landmarks[0][[263, 362, 384, 385, 386, 387, 388, 398, 466]], axis=0)
                    mp_eye_center = (mp_left_eye + mp_right_eye) / 2
                    
                    # dlib的眼睛特征点索引 (36-41是左眼，42-47是右眼)
                    dlib_left_eye = np.mean(dlib_landmarks[0][36:42], axis=0)
                    dlib_right_eye = np.mean(dlib_landmarks[0][42:48], axis=0)
                    dlib_eye_center = (dlib_left_eye + dlib_right_eye) / 2
                    
                    # 计算两个眼睛中心的距离
                    eye_center_distance = np.linalg.norm(mp_eye_center - dlib_eye_center)
                    
                    # 如果距离小于阈值，使用MediaPipe结果
                    if eye_center_distance < 30:  # 30像素阈值可调整
                        logger.info(f"MediaPipe和dlib检测结果一致(距离:{eye_center_distance:.2f}像素)，使用MediaPipe结果")
                        final_landmarks = mediapipe_landmarks
                    else:
                        logger.warning(f"MediaPipe和dlib检测结果不一致(距离:{eye_center_distance:.2f}像素)，使用dlib结果")
                        final_landmarks = self._convert_dlib_to_mediapipe_format(dlib_landmarks, image.shape)
                else:
                    # 只有MediaPipe检测到了
                    logger.info("只有MediaPipe检测到人脸，使用MediaPipe结果")
                    final_landmarks = mediapipe_landmarks
            elif dlib_landmarks and len(dlib_landmarks) > 0:
                # 只有dlib检测到了
                logger.info("只有dlib检测到人脸，使用dlib结果")
                final_landmarks = self._convert_dlib_to_mediapipe_format(dlib_landmarks, image.shape)
            else:
                # 两者都没检测到
                logger.warning("MediaPipe和dlib均未检测到人脸")
                return None
            
            # 存入缓存
            self.landmarks_cache[image_hash] = final_landmarks
            
            # 限制缓存大小
            if len(self.landmarks_cache) > 50:
                self.landmarks_cache.pop(next(iter(self.landmarks_cache)))
            
            return final_landmarks
    
    def _convert_dlib_to_mediapipe_format(self, dlib_landmarks: List[np.ndarray], image_shape: tuple) -> List[np.ndarray]:
        """将dlib的68点特征转换为与MediaPipe兼容的格式"""
        # 这是一个简化的转换，实际上MediaPipe有468个特征点
        # 我们只关心眼睛和鼻子的位置，所以只需要确保这些关键点被正确映射
        
        result = []
        for dlib_points in dlib_landmarks:
            # 创建一个与MediaPipe特征点数量相同的数组
            mp_points = np.zeros((468, 2), dtype=np.float32)
            
            # 映射关键点 (这些映射是近似的)
            # 左眼
            mp_points[33] = dlib_points[36]  # 左眼外角
            mp_points[133] = dlib_points[39]  # 左眼内角
            mp_points[157] = dlib_points[37]
            mp_points[158] = dlib_points[38]
            mp_points[159] = dlib_points[39]
            mp_points[160] = dlib_points[40]
            mp_points[161] = dlib_points[41]
            mp_points[173] = dlib_points[38]
            mp_points[246] = dlib_points[37]
            
            # 右眼
            mp_points[263] = dlib_points[45]  # 右眼外角
            mp_points[362] = dlib_points[42]  # 右眼内角
            mp_points[384] = dlib_points[43]
            mp_points[385] = dlib_points[44]
            mp_points[386] = dlib_points[45]
            mp_points[387] = dlib_points[46]
            mp_points[388] = dlib_points[47]
            mp_points[398] = dlib_points[44]
            mp_points[466] = dlib_points[43]
            
            # 鼻尖
            mp_points[1] = dlib_points[30]
            
            # 下巴
            mp_points[152] = dlib_points[8]
            
            result.append(mp_points)
        
        return result

    def process_mask(self, mask: torch.Tensor, feather: int = 0) -> np.ndarray:
        """处理mask，添加羽化效果"""
        with PerformanceTimer("处理遮罩"):
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8)
            
            if feather > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather, feather))
                mask = cv2.dilate(mask, kernel)
                mask = cv2.GaussianBlur(mask, (feather//2*2+1, feather//2*2+1), 0)
            return mask
    
    def get_face_info(self, landmarks: np.ndarray) -> Dict:
        """获取单个人脸的信息"""
        with PerformanceTimer("计算面部信息"):
            # 使用更多的眼睛周围特征点来提高定位精度
            # 左眼特征点（MediaPipe的眼睛轮廓点）
            left_eye_points = landmarks[[33, 133, 157, 158, 159, 160, 161, 173, 246]]
            # 右眼特征点
            right_eye_points = landmarks[[263, 362, 384, 385, 386, 387, 388, 398, 466]]
            
            # 计算眼睛中心（使用多个点的平均值）
            left_eye = np.mean(left_eye_points, axis=0)
            right_eye = np.mean(right_eye_points, axis=0)
            
            # 鼻尖和下巴
            nose_tip = landmarks[1]  # 鼻尖
            chin = landmarks[152]    # 下巴
            
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            face_angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            
            # 记录关键点位置用于调试
            logger.info(f"左眼位置(多点平均): {left_eye}, 右眼位置(多点平均): {right_eye}")
            logger.info(f"眼睛中心: {eye_center}, 面部角度: {face_angle}°")
            
            return {
                'eye_center': eye_center,
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose_tip': nose_tip,
                'chin': chin,
                'face_angle': face_angle,
                'eye_distance': np.linalg.norm(eye_vector)
            }
    
    def align_and_paste_face(self, target: torch.Tensor,
                           source: torch.Tensor,
                           mask: torch.Tensor,
                           feather: int = 0,
                           size: float = 1.0,
                           move_x: int = 0,
                           move_y: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理单个人脸的对齐和替换"""
        try:
            with PerformanceTimer("总处理时间"):
                # 优化哈希计算
                target_np = target[0].cpu().numpy()
                source_np = source[0].cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                # 计算输入哈希
                th, tw = target_np.shape[:2]
                sh, sw = source_np.shape[:2]
                
                # 使用低分辨率计算哈希
                t_down = target_np[::16, ::16, :]
                s_down = source_np[::16, ::16, :]
                m_down = mask_np[0, ::16, ::16]
                
                input_hash = hash((
                    hash(t_down.tobytes()),
                    hash(s_down.tobytes()),
                    hash(m_down.tobytes()),
                    feather, size, move_x, move_y
                ))
                
                # 检查缓存
                if input_hash in self.last_processed_images:
                    logger.info("使用缓存的处理结果")
                    return self.last_processed_images[input_hash]
                
                # 数据准备
                with PerformanceTimer("数据准备"):
                    # 转换为8位整数格式
                    source_np = (source_np * 255).astype(np.uint8)
                    target_np = (target_np * 255).astype(np.uint8)
                    mask_np = self.process_mask(mask, feather)
                
                # 获取特征点
                with PerformanceTimer("特征点提取"):
                    src_landmarks_list = self.get_face_landmarks(source_np)
                    target_landmarks_list = self.get_face_landmarks(target_np)
                    
                    if not src_landmarks_list:
                        logger.error("源图像未检测到人脸，请确保源图像中有清晰的人脸")
                        empty_mask = torch.zeros((1, target.shape[2], target.shape[3]), dtype=torch.float32)
                        return (target.cpu(), empty_mask, empty_mask)
                        
                    if not target_landmarks_list:
                        logger.error("目标图像未检测到人脸，请确保目标图像中有清晰的人脸")
                        empty_mask = torch.zeros((1, target.shape[2], target.shape[3]), dtype=torch.float32)
                        return (target.cpu(), empty_mask, empty_mask)
                    
                    src_landmarks = src_landmarks_list[0]
                    target_landmarks = target_landmarks_list[0]
                
                # 计算面部信息
                src_info = self.get_face_info(src_landmarks)
                tgt_info = self.get_face_info(target_landmarks)
                
                # 计算变换参数
                with PerformanceTimer("图像变换"):
                    # 只使用眼睛中心进行对齐，不进行扭曲变换
                    # 计算源图像和目标图像的眼睛中心
                    src_eye_center = src_info['eye_center']
                    tgt_eye_center = tgt_info['eye_center']
                    
                    # 计算缩放比例 - 基于眼睛距离
                    scale = tgt_info['eye_distance'] / src_info['eye_distance'] * size
                    
                    # 计算旋转角度 - 基于眼睛连线的角度差
                    angle_diff = tgt_info['face_angle'] - src_info['face_angle']
                    
                    logger.info(f"变换参数 - 角度差: {angle_diff:.2f}°, 缩放比例: {scale:.2f}")
                    
                    # 创建仿射变换矩阵 - 先旋转和缩放
                    matrix = cv2.getRotationMatrix2D(
                        tuple(src_eye_center),  # 以源图像眼睛中心为旋转中心
                        -angle_diff,            # 旋转角度
                        scale                   # 缩放比例
                    )
                    
                    # 添加平移分量 - 使眼睛中心对齐
                    matrix[0, 2] += tgt_eye_center[0] - src_eye_center[0]  # X轴平移
                    matrix[1, 2] += tgt_eye_center[1] - src_eye_center[1]  # Y轴平移
                    
                    # 应用额外的移动
                    if move_x != 0 or move_y != 0:
                        matrix[0, 2] += move_x * 4  # X轴额外平移
                        matrix[1, 2] -= move_y * 4  # Y轴额外平移
                    
                    # 执行变换
                    img_transformed = cv2.warpAffine(
                        source_np, matrix, (target_np.shape[1], target_np.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    
                    mask_transformed = cv2.warpAffine(
                        mask_np, matrix, (target_np.shape[1], target_np.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    
                    # 保存原始mask（未羽化）
                    raw_mask_transformed = mask_transformed.copy()
                
                # 创建混合mask
                with PerformanceTimer("图像混合"):
                    if feather > 0:
                        mask_blur = cv2.GaussianBlur(mask_transformed, (feather*2+1, feather*2+1), 0)
                    else:
                        mask_blur = mask_transformed
                    
                    # 检查mask是否有效
                    if np.max(mask_blur) < 10:  # 如果mask几乎全黑
                        logger.warning("变换后的mask几乎为空，可能是人脸定位不准确导致")
                    
                    # 使用向量化操作
                    alpha = mask_blur[..., None].astype(np.float32) / 255.0
                    
                    # 合成图像
                    result = target_np.astype(np.float32) * (1 - alpha) + img_transformed.astype(np.float32) * alpha
                    result = result.astype(np.uint8)
                    
                # 转换结果
                with PerformanceTimer("结果转换"):
                    result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
                    mask_tensor = torch.from_numpy(mask_blur.astype(np.float32) / 255.0).unsqueeze(0)
                    raw_mask_tensor = torch.from_numpy(raw_mask_transformed.astype(np.float32) / 255.0).unsqueeze(0)
                
                # 缓存处理结果
                output = (result_tensor, mask_tensor, raw_mask_tensor)
                self.last_processed_images[input_hash] = output
                
                # 限制缓存大小
                if len(self.last_processed_images) > 30:
                    self.last_processed_images.pop(next(iter(self.last_processed_images)))
                
                return output
            
        except Exception as e:
            logger.error(f"人脸处理失败: {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            empty_mask = torch.zeros((1, target.shape[2], target.shape[3]), dtype=torch.float32)
            return (target.cpu(), empty_mask, empty_mask)

NODE_CLASS_MAPPINGS = {"FaceAlignSingle": FaceAlignSingle}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignSingle": "Face Align Single @ CHAOS"}