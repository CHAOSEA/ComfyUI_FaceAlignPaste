from typing import Dict, Tuple
from .logging_utils import logger

class PersonParams:
    @staticmethod
    def get_person_params(mode, face_info) -> Dict[str, float]:
        """获取人物比例参数 - 基于人脸分析而非性别假设"""
        # 默认参数 - 适用于未知性别
        params = {
            "head_ratio": 0.18,           # 头部占画布高度的比例
            "eye_position_ratio": 0.30    # 眼睛位于画布高度的比例
        }
        
        # 根据模式调整基础参数 - 基于参考图片
        if mode == "portrait":
            # 肖像模式 - 参考图1
            params["head_ratio"] = 0.28    # 头部比例约占28%
            params["eye_position_ratio"] = 0.33  # 眼睛位置约在1/3处
        elif mode == "half_body":
            # 半身模式 - 参考图2
            params["head_ratio"] = 0.18    # 头部比例约占18%
            params["eye_position_ratio"] = 0.25  # 眼睛位置约在1/4处
        elif mode == "full_body":
            # 全身模式 - 参考图3
            params["head_ratio"] = 0.10    # 头部比例约占10%
            params["eye_position_ratio"] = 0.12  # 眼睛位置约在1/8处
        
        # 根据性别微调参数 - 加强男女差异
        if face_info["gender"] == "male":
            # 男性通常头部略大，眼睛位置略高
            params["head_ratio"] *= 1.05
            params["eye_position_ratio"] *= 0.93  # 更强的上移效果
        elif face_info["gender"] == "female":
            # 女性通常头部略小，眼睛位置略低
            params["head_ratio"] *= 0.95
            params["eye_position_ratio"] *= 1.07  # 更强的下移效果
        
        logger.info(f"人物参数: 性别={face_info['gender']}, 头部比例={params['head_ratio']:.3f}, 眼睛位置={params['eye_position_ratio']:.3f}")
        
        return params
    
    @staticmethod
    def calculate_dimensions(mode, custom_width, custom_height) -> Tuple[int, int]:
        """智能尺寸计算系统 - 基于参考图片"""
        # 注意：宽度在前，高度在后
        mode_params = {
            "portrait": (1024, 720),     # 肖像模式 - 宽1024，高720
            "half_body": (1024, 1536),   # 半身模式
            "full_body": (1280, 1600),   # 全身模式
            "custom": (custom_width, custom_height)  # 自定义模式
        }
        
        width, height = mode_params[mode]
        logger.info(f"目标画布尺寸: {width}×{height}")
        return width, height