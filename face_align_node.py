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
logger = logging.getLogger("FaceAlignDouble")

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

class FaceAlignDouble:
    """Face alignment and pasting node for ComfyUI (Double Face Version)"""
    
    def __init__(self):
        with PerformanceTimer("初始化人脸检测模型"):
            # 使用MediaPipe Face Mesh作为主要检测器
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,  # 检测两个人脸
                refine_landmarks=True,  # 使用精细特征点以提高准确性
                min_detection_confidence=0.5
            )
            
            # 初始化dlib检测器作为备用
            if has_dlib:
                # 获取模型文件路径 - 使用models子目录
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
            },
            "optional": {
                # 左侧人脸参数
                "image_L": ("IMAGE",),
                "mask_L": ("MASK",),
                "feather_L": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "size_L": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05
                }),
                "move_x_L": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "move_y_L": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                # 右侧人脸参数
                "image_R": ("IMAGE",),
                "mask_R": ("MASK",),
                "feather_R": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "size_R": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05
                }),
                "move_x_R": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "move_y_R": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask_L", "mask_R", "mask_combined", "unfeathered_mask_combined")
    FUNCTION = "align_and_paste_faces"
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
                logger.info(f"MediaPipe成功检测到{len(mediapipe_landmarks)}个人脸")
            else:
                logger.warning("MediaPipe未检测到人脸，尝试使用dlib")
            
            # 2. 如果MediaPipe失败或者我们有dlib，也用dlib检测
            dlib_landmarks = None
            if self.dlib_detector is not None and (mediapipe_landmarks is None or len(mediapipe_landmarks) < 2):
                # 转换为灰度图像用于dlib
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 检测人脸
                dlib_faces = self.dlib_detector(gray)
                
                if len(dlib_faces) > 0:
                    # 获取人脸特征点
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
            final_landmarks = []
            
            # 如果MediaPipe检测到了人脸
            if mediapipe_landmarks and len(mediapipe_landmarks) > 0:
                # 如果dlib也检测到了人脸，检查一致性
                if dlib_landmarks and len(dlib_landmarks) > 0:
                    # 对每个MediaPipe人脸，找到最匹配的dlib人脸
                    for mp_face in mediapipe_landmarks:
                        mp_left_eye = np.mean(mp_face[[33, 133, 157, 158, 159, 160, 161, 173, 246]], axis=0)
                        mp_right_eye = np.mean(mp_face[[263, 362, 384, 385, 386, 387, 388, 398, 466]], axis=0)
                        mp_eye_center = (mp_left_eye + mp_right_eye) / 2
                        
                        best_match = None
                        min_distance = float('inf')
                        
                        for dlib_face in dlib_landmarks:
                            dlib_left_eye = np.mean(dlib_face[36:42], axis=0)
                            dlib_right_eye = np.mean(dlib_face[42:48], axis=0)
                            dlib_eye_center = (dlib_left_eye + dlib_right_eye) / 2
                            
                            distance = np.linalg.norm(mp_eye_center - dlib_eye_center)
                            if distance < min_distance:
                                min_distance = distance
                                best_match = dlib_face
                        
                        # 如果找到匹配且距离小于阈值，使用MediaPipe结果
                        if min_distance < 30:
                            logger.info(f"MediaPipe和dlib检测结果一致(距离:{min_distance:.2f}像素)，使用MediaPipe结果")
                            final_landmarks.append(mp_face)
                        else:
                            # 否则使用dlib结果
                            logger.info(f"MediaPipe和dlib检测结果不一致，使用dlib结果")
                            final_landmarks.append(self._convert_dlib_to_mediapipe_format(best_match, image.shape))
                else:
                    # 只有MediaPipe检测到了
                    logger.info("只有MediaPipe检测到人脸，使用MediaPipe结果")
                    final_landmarks = mediapipe_landmarks
            
            # 如果最终结果少于dlib检测到的人脸数，添加dlib检测到但MediaPipe未检测到的人脸
            if dlib_landmarks and len(final_landmarks) < len(dlib_landmarks):
                for dlib_face in dlib_landmarks:
                    dlib_left_eye = np.mean(dlib_face[36:42], axis=0)
                    dlib_right_eye = np.mean(dlib_face[42:48], axis=0)
                    dlib_eye_center = (dlib_left_eye + dlib_right_eye) / 2
                    
                    # 检查是否已经添加了这个人脸
                    is_new_face = True
                    for face in final_landmarks:
                        face_left_eye = np.mean(face[[33, 133, 157, 158, 159, 160, 161, 173, 246]], axis=0)
                        face_right_eye = np.mean(face[[263, 362, 384, 385, 386, 387, 388, 398, 466]], axis=0)
                        face_eye_center = (face_left_eye + face_right_eye) / 2
                        
                        if np.linalg.norm(face_eye_center - dlib_eye_center) < 30:
                            is_new_face = False
                            break
                    
                    if is_new_face:
                        logger.info("添加dlib检测到的额外人脸")
                        final_landmarks.append(self._convert_dlib_to_mediapipe_format(dlib_face, image.shape))
            
            # 如果没有检测到人脸，返回None
            if not final_landmarks:
                logger.warning("MediaPipe和dlib均未检测到人脸")
                return None
            
            # 存入缓存
            self.landmarks_cache[image_hash] = final_landmarks
            
            # 限制缓存大小
            if len(self.landmarks_cache) > 50:
                self.landmarks_cache.pop(next(iter(self.landmarks_cache)))
            
            return final_landmarks
    
    def _convert_dlib_to_mediapipe_format(self, dlib_points: np.ndarray, image_shape: tuple) -> np.ndarray:
        """将dlib的68点特征转换为与MediaPipe兼容的格式"""
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
        
        return mp_points

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
            # MediaPipe特征点索引与face_alignment不同
            # 左眼: 33个点中的133和33
            # 右眼: 33个点中的362和263
            left_eye = np.mean(landmarks[[33, 133]], axis=0)
            right_eye = np.mean(landmarks[[263, 362]], axis=0)
            nose_tip = landmarks[1]  # 鼻尖
            chin = landmarks[152]    # 下巴
            
            eye_center = (left_eye + right_eye) / 2
            eye_vector = right_eye - left_eye
            face_angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            
            return {
                'eye_center': eye_center,
                'nose_tip': nose_tip,
                'chin': chin,
                'face_angle': face_angle,
                'eye_distance': np.linalg.norm(eye_vector)
            }
    
    def sort_faces_by_position(self, landmarks_list: List[np.ndarray]) -> List[Tuple[float, np.ndarray]]:
        """按照水平位置排序人脸"""
        face_positions = []
        for landmarks in landmarks_list:
            # 使用眼睛区域的平均x坐标作为人脸位置
            face_center_x = np.mean(landmarks[[33, 133, 263, 362]], axis=0)[0]
            face_positions.append((face_center_x, landmarks))
        # 按x坐标从左到右排序
        return sorted(face_positions, key=lambda x: x[0])

    def align_single_face(self, source_image: torch.Tensor, source_mask: torch.Tensor,
                         target_image: np.ndarray, target_landmarks: np.ndarray,
                         feather_amount: int, size_adjust: float,
                         move_to_x: int = 0, move_to_y: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理单个人脸的对齐和替换，返回结果图像和对应的mask"""
        try:
            # 转换数据格式
            source_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
            mask_np = self.process_mask(source_mask, 0)  # 不应用羽化，保留原始mask
            
            # 获取源图像特征点
            src_landmarks_list = self.get_face_landmarks(source_np)
            if not src_landmarks_list:
                logger.error("源图像未检测到人脸")
                return target_image, np.zeros_like(target_image[:,:,0]), np.zeros_like(target_image[:,:,0])
            
            src_landmarks = src_landmarks_list[0]
            
            # 计算面部信息
            src_info = self.get_face_info(src_landmarks)
            tgt_info = self.get_face_info(target_landmarks)
            
            # 计算变换参数
            with PerformanceTimer("图像变换"):
                angle_diff = tgt_info['face_angle'] - src_info['face_angle']
                scale = tgt_info['eye_distance'] / src_info['eye_distance'] * size_adjust
                
                # 创建仿射变换矩阵
                matrix = cv2.getRotationMatrix2D(
                    tuple(src_info['eye_center']),
                    -angle_diff,
                    scale
                )
                
                # 添加平移分量
                matrix[0, 2] += tgt_info['eye_center'][0] - src_info['eye_center'][0]
                matrix[1, 2] += tgt_info['eye_center'][1] - src_info['eye_center'][1]
                
                # 执行变换
                img_transformed = cv2.warpAffine(
                    source_np, matrix, (target_image.shape[1], target_image.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                mask_transformed = cv2.warpAffine(
                    mask_np, matrix, (target_image.shape[1], target_image.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                
                # 保存原始mask（未羽化）
                raw_mask_transformed = mask_transformed.copy()
                
                # 应用额外的移动
                if move_to_x != 0 or move_to_y != 0:
                    M = np.float32([[1, 0, move_to_x * 4], [0, 1, -move_to_y * 4]])
                    img_transformed = cv2.warpAffine(img_transformed, M, (img_transformed.shape[1], img_transformed.shape[0]))
                    mask_transformed = cv2.warpAffine(mask_transformed, M, (mask_transformed.shape[1], mask_transformed.shape[0]))
                    raw_mask_transformed = cv2.warpAffine(raw_mask_transformed, M, (raw_mask_transformed.shape[1], raw_mask_transformed.shape[0]))
            
            # 创建混合mask
            with PerformanceTimer("图像混合"):
                if feather_amount > 0:
                    mask_blur = cv2.GaussianBlur(mask_transformed, (feather_amount*2+1, feather_amount*2+1), 0)
                else:
                    mask_blur = mask_transformed
                
                # 使用向量化操作
                alpha = mask_blur[..., None].astype(np.float32) / 255.0
                
                # 合成图像
                result = target_image.astype(np.float32) * (1 - alpha) + img_transformed.astype(np.float32) * alpha
                result = result.astype(np.uint8)
            
            return result, mask_blur, raw_mask_transformed
            
        except Exception as e:
            logger.error(f"单个人脸处理失败: {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return target_image, np.zeros_like(target_image[:,:,0]), np.zeros_like(target_image[:,:,0])

    def align_and_paste_faces(self, target: torch.Tensor,
                            image_L: Optional[torch.Tensor] = None,
                            mask_L: Optional[torch.Tensor] = None,
                            feather_L: int = 0,
                            size_L: float = 1.0,
                            move_x_L: int = 0,
                            move_y_L: int = 0,
                            image_R: Optional[torch.Tensor] = None,
                            mask_R: Optional[torch.Tensor] = None,
                            feather_R: int = 0,
                            size_R: float = 1.0,
                            move_x_R: int = 0,
                            move_y_R: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理一个或两个人脸的对齐和替换"""
        try:
            with PerformanceTimer("总处理时间"):
                # 优化哈希计算
                target_np = target[0].cpu().numpy()
                
                # 计算输入哈希
                t_down = target_np[::16, ::16, :]
                
                # 构建哈希值
                hash_components = [hash(t_down.tobytes())]
                
                if image_L is not None and mask_L is not None:
                    l_down = image_L[0].cpu().numpy()[::16, ::16, :]
                    ml_down = mask_L.cpu().numpy()[0, ::16, ::16]
                    hash_components.extend([
                        hash(l_down.tobytes()),
                        hash(ml_down.tobytes()),
                        feather_L, size_L, move_x_L, move_y_L
                    ])
                
                if image_R is not None and mask_R is not None:
                    r_down = image_R[0].cpu().numpy()[::16, ::16, :]
                    mr_down = mask_R.cpu().numpy()[0, ::16, ::16]
                    hash_components.extend([
                        hash(r_down.tobytes()),
                        hash(mr_down.tobytes()),
                        feather_R, size_R, move_x_R, move_y_R
                    ])
                
                input_hash = hash(tuple(hash_components))
                
                # 检查缓存
                if input_hash in self.last_processed_images:
                    logger.info("使用缓存的处理结果")
                    return self.last_processed_images[input_hash]
                
                # 数据准备
                with PerformanceTimer("数据准备"):
                    # 转换为8位整数格式
                    target_np = (target_np * 255).astype(np.uint8)
                
                # 获取目标图像中的人脸特征点
                with PerformanceTimer("特征点提取"):
                    target_landmarks_list = self.get_face_landmarks(target_np)
                    
                    if not target_landmarks_list or len(target_landmarks_list) == 0:
                        logger.error("目标图像未检测到人脸")
                        empty_mask = torch.zeros((1, target.shape[2], target.shape[3]), dtype=torch.float32)
                        return (target.cpu(), empty_mask, empty_mask, empty_mask, empty_mask, empty_mask)
                    
                    # 按照水平位置排序人脸
                    sorted_faces = self.sort_faces_by_position(target_landmarks_list)
                    logger.info(f"检测到 {len(sorted_faces)} 个人脸")
                
                # 初始化结果
                result_image = target_np.copy()
                h, w = result_image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                mask_left = empty_mask.copy()
                raw_mask_left = empty_mask.copy()
                mask_right = empty_mask.copy()
                raw_mask_right = empty_mask.copy()
                
                # 处理左侧人脸（如果有输入和目标脸）
                if len(sorted_faces) > 0 and image_L is not None and mask_L is not None:
                    logger.info("处理左侧人脸")
                    result_image, mask_left, raw_mask_left = self.align_single_face(
                        image_L, mask_L, result_image,
                        sorted_faces[0][1], feather_L, size_L,
                        move_x_L, move_y_L
                    )
                
                # 处理右侧人脸（如果有输入和目标脸）
                if len(sorted_faces) > 1 and image_R is not None and mask_R is not None:
                    logger.info("处理右侧人脸")
                    result_image, mask_right, raw_mask_right = self.align_single_face(
                        image_R, mask_R, result_image,
                        sorted_faces[1][1], feather_R, size_R,
                        move_x_R, move_y_R
                    )
                
                # 合并mask（羽化后的和未羽化的）
                mask_combined = np.maximum(mask_left, mask_right)
                raw_mask_combined = np.maximum(raw_mask_left, raw_mask_right)
                
                # 转换结果
                with PerformanceTimer("结果转换"):
                    result_tensor = torch.from_numpy(result_image.astype(np.float32) / 255.0).unsqueeze(0)
                    mask_left_tensor = torch.from_numpy(mask_left.astype(np.float32) / 255.0).unsqueeze(0)
                    mask_right_tensor = torch.from_numpy(mask_right.astype(np.float32) / 255.0).unsqueeze(0)
                    mask_combined_tensor = torch.from_numpy(mask_combined.astype(np.float32) / 255.0).unsqueeze(0)
                    raw_mask_combined_tensor = torch.from_numpy(raw_mask_combined.astype(np.float32) / 255.0).unsqueeze(0)
                
                # 缓存处理结果
                output = (result_tensor, mask_left_tensor, mask_right_tensor, 
                          mask_combined_tensor, raw_mask_combined_tensor)
                self.last_processed_images[input_hash] = output
                
                # 限制缓存大小
                if len(self.last_processed_images) > 20:
                    self.last_processed_images.pop(next(iter(self.last_processed_images)))
                
                return output
            
        except Exception as e:
            logger.error(f"人脸处理失败: {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            empty_mask = torch.zeros((1, target.shape[2], target.shape[3]), dtype=torch.float32)
            return (target.cpu(), empty_mask, empty_mask, empty_mask, empty_mask, empty_mask)

NODE_CLASS_MAPPINGS = {"FaceAlignDouble": FaceAlignDouble}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignDouble": "Face Align Double @ CHAOS"}