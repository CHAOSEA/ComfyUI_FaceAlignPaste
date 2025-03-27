import numpy as np
import cv2
import torch
import os
from typing import Tuple
import logging

logger = logging.getLogger("ComfyUI_FaceAlignPaste")

class GenderRecognitionNode:
    """性别识别节点 - 使用InsightFace"""
    
    # 静态变量用于存储InsightFace分析器
    _face_analyzer = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("gender", "gender_int",)
    CATEGORY = "image/face"
    FUNCTION = "recognize_gender"

    def __init__(self):
        # 不再需要加载Caffe模型
        pass

    def recognize_gender(self, input_image) -> Tuple[str, int]:
        # 预处理输入图像
        image = (input_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 延迟加载InsightFace模型
        if GenderRecognitionNode._face_analyzer is None:
            GenderRecognitionNode._face_analyzer = self._load_insightface()
            if GenderRecognitionNode._face_analyzer is None:
                logger.warning("InsightFace模型加载失败，无法识别性别")
                return ("Unknown", -1)
        
        try:
            # 分析人脸
            faces = GenderRecognitionNode._face_analyzer.get(rgb_image)
            
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
                
                if max_face is not None and hasattr(max_face, 'gender') and max_face.gender is not None:
                    gender_idx = int(max_face.gender)
                    gender_str = "Male" if gender_idx == 1 else "Female"
                    gender_int = 1 if gender_idx == 1 else 0
                    logger.info(f"InsightFace检测性别: {gender_str}")
                    return (gender_str, gender_int)
        except Exception as e:
            logger.warning(f"InsightFace性别识别失败: {str(e)}")
        
        return ("Unknown", -1)
    
    def _load_insightface(self):
        """加载InsightFace模型"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # 获取ComfyUI根目录（使用相对路径）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfyui_root = os.path.dirname(os.path.dirname(current_dir))
            
            # 设置模型路径为相对路径
            models_dir = os.path.join(comfyui_root, "models", "insightface")
            
            # 设置InsightFace模型路径
            os.environ['INSIGHTFACE_HOME'] = models_dir
            
            # 创建分析器 - 使用更小的检测尺寸以提高速度
            face_analyzer = FaceAnalysis(
                name="buffalo_l",
                root=models_dir,
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'genderage']  # 只加载必要的模块
            )
            
            # 准备模型 - 使用更小的检测尺寸
            face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
            
            logger.info("InsightFace模型加载成功")
            return face_analyzer
        except Exception as e:
            logger.error(f"InsightFace模型加载失败: {str(e)}")
            return None