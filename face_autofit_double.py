import torch
import numpy as np
import cv2  # 添加cv2导入
from typing import Tuple, Dict, Any

from .utils.logging_utils import logger, Timer
from .utils.face_detection import FaceDetector
from .utils.image_processing import ImageProcessor
from .utils.person_params import PersonParams

class FaceAutoFitDouble:
    """智能双人人脸适配节点 - 增强版"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_L": ("IMAGE",),  # 左侧人物图像
                "mask_L": ("MASK",),    # 左侧人物蒙版
                "image_R": ("IMAGE",),  # 右侧人物图像
                "mask_R": ("MASK",),    # 右侧人物蒙版
                "mode": (["portrait", "half_body", "full_body", "custom"], {"default": "portrait"}),
                "background": (["white", "gray"], {"default": "white"}),
                "move_L_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "move_L_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "move_R_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "move_R_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "face_size_L": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "face_size_R": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "angle_L": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "angle_R": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "detection_method": (["insightface", "auto", "opencv", "mediapipe", "dlib", "none"], {"default": "insightface"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask_L", "mask_R", "mask_combined", "gender_L", "gender_R", "mode_code")
    CATEGORY = "image/face"
    FUNCTION = "process"

    def process(self, image_L, mask_L, image_R, mask_R, mode, background, 
               move_L_x, move_L_y, move_R_x, move_R_y, 
               face_size_L, face_size_R, angle_L=0.0, angle_R=0.0, 
               detection_method="insightface", custom_width=1024, custom_height=1024):
        try:
            with Timer("总处理时间"):
                # 保存当前模式和检测方法，供其他方法使用
                self.current_mode = mode
                self.detection_method = detection_method
                
                # 计算目标尺寸和背景颜色
                target_w, target_h = PersonParams.calculate_dimensions(mode, custom_width, custom_height)
                bg_color = ImageProcessor.get_background_color(background)
                
                # 处理左侧人物 -------------------------------------------------
                with Timer("处理左侧人物"):
                    # 核心处理流程
                    np_face_L, np_mask_L = ImageProcessor.prepare_inputs(image_L, mask_L)
                    
                    # 检测人脸中心点，用于旋转和缩放时保持中心不变
                    with Timer("左侧人脸分析"):
                        face_info_L = FaceDetector.analyze_face(np_face_L, np_mask_L, detection_method)
                        face_center_L = face_info_L["eyes_center"] if face_info_L["eyes_center"] is not None else (np_face_L.shape[1]//2, np_face_L.shape[0]//2)
                    
                    # 获取人物比例参数
                    params_L = PersonParams.get_person_params(mode, face_info_L)
                    
                    # 智能缩放
                    with Timer("左侧智能缩放"):
                        scaled_face_L, scaled_mask_L, face_center_scaled_L = ImageProcessor.smart_scaling(
                            np_face_L, np_mask_L, target_h, params_L["head_ratio"], face_size_L, face_center_L, face_info_L
                        )
                    
                    # 应用旋转
                    if angle_L != 0:
                        with Timer("左侧图像旋转"):
                            scaled_face_L, scaled_mask_L, face_center_scaled_L = ImageProcessor.rotate_image(
                                scaled_face_L, scaled_mask_L, angle_L, bg_color, face_center_scaled_L
                            )
                    
                    # 精确定位
                    with Timer("左侧精确定位"):
                        final_face_L, final_mask_L, final_face_center_L = ImageProcessor.apply_positioning(
                            scaled_face_L, scaled_mask_L, target_w, target_h, bg_color, 
                            move_L_x, move_L_y, face_center_scaled_L, params_L["eye_position_ratio"],
                            face_info_L["gender"], "left"
                        )
                    
                    # 检查人物是否脱离画布
                    ImageProcessor.check_boundary(final_mask_L, "左侧人物")
                    
                    # 获取性别信息
                    is_male_L = face_info_L["gender"] == "male"
                    gender_code_L = 1 if is_male_L else 0
                
                # 处理右侧人物 -------------------------------------------------
                with Timer("处理右侧人物"):
                    # 核心处理流程
                    np_face_R, np_mask_R = ImageProcessor.prepare_inputs(image_R, mask_R)
                    
                    # 检测人脸中心点，用于旋转和缩放时保持中心不变
                    with Timer("右侧人脸分析"):
                        face_info_R = FaceDetector.analyze_face(np_face_R, np_mask_R, detection_method)
                        face_center_R = face_info_R["eyes_center"] if face_info_R["eyes_center"] is not None else (np_face_R.shape[1]//2, np_face_R.shape[0]//2)
                    
                    # 获取人物比例参数
                    params_R = PersonParams.get_person_params(mode, face_info_R)
                    
                    # 关键修改：使用左侧人脸的缩放信息来调整右侧人脸
                    # 如果左侧人脸检测成功，使用相同的目标人脸高度
                    if "target_face_height" in face_info_L and face_info_L["target_face_height"] is not None:
                        if face_info_R["face_height"] is not None:
                            # 计算右侧人脸应该使用的缩放比例
                            right_scale = (face_info_L["target_face_height"] / face_info_R["face_height"]) * (face_size_R / face_size_L)
                            logger.info(f"基于左侧人脸高度调整右侧人脸缩放比例: {right_scale:.3f}")
                            
                            # 将这个缩放比例应用到右侧人脸
                            face_info_R["forced_scale"] = right_scale
                    
                    # 智能缩放
                    with Timer("右侧智能缩放"):
                        # 如果有强制缩放比例，使用它
                        if "forced_scale" in face_info_R and face_info_R["forced_scale"] is not None:
                            # 直接应用缩放比例
                            scale_factor = face_info_R["forced_scale"]
                            new_h = int(np_face_R.shape[0] * scale_factor)
                            new_w = int(np_face_R.shape[1] * scale_factor)
                            
                            # 确保尺寸至少为1像素
                            new_h = max(1, new_h)
                            new_w = max(1, new_w)
                            
                            # 使用高质量插值方法进行缩放
                            scaled_face_R = cv2.resize(np_face_R, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                            scaled_mask_R = cv2.resize(np_mask_R, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            
                            # 计算缩放后的人脸中心点
                            if face_info_R["eyes_center"] is not None:
                                eyes_x, eyes_y = face_info_R["eyes_center"]
                                face_center_scaled_R = (int(eyes_x * scale_factor), int(eyes_y * scale_factor))
                            else:
                                cx, cy = face_center_R
                                face_center_scaled_R = (int(cx * scale_factor), int(cy * scale_factor))
                        else:
                            # 使用标准缩放方法
                            scaled_face_R, scaled_mask_R, face_center_scaled_R = ImageProcessor.smart_scaling(
                                np_face_R, np_mask_R, target_h, params_R["head_ratio"], face_size_R, face_center_R, face_info_R
                            )
                    
                    # 应用旋转
                    if angle_R != 0:
                        with Timer("右侧图像旋转"):
                            scaled_face_R, scaled_mask_R, face_center_scaled_R = ImageProcessor.rotate_image(
                                scaled_face_R, scaled_mask_R, angle_R, bg_color, face_center_scaled_R
                            )
                    
                    # 精确定位
                    with Timer("右侧精确定位"):
                        final_face_R, final_mask_R, final_face_center_R = ImageProcessor.apply_positioning(
                            scaled_face_R, scaled_mask_R, target_w, target_h, bg_color, 
                            move_R_x, move_R_y, face_center_scaled_R, params_R["eye_position_ratio"],
                            face_info_R["gender"], "right"
                        )
                    
                    # 检查人物是否脱离画布
                    ImageProcessor.check_boundary(final_mask_R, "右侧人物")
                    
                    # 获取性别信息
                    is_male_R = face_info_R["gender"] == "male"
                    gender_code_R = 1 if is_male_R else 0
                
                # 计算mode_code
                # 计算性别组合编码 (0-3)
                if not is_male_L and not is_male_R:
                    gender_code = 0  # 女性组合
                elif not is_male_L and is_male_R:
                    gender_code = 1  # 左女右男
                elif is_male_L and not is_male_R:
                    gender_code = 2  # 左男右女
                else:  # is_male_L and is_male_R
                    gender_code = 3  # 男性组合
                
                # 计算模式编码 (0-3)
                if mode == "portrait":
                    mode_value = 0
                elif mode == "half_body":
                    mode_value = 1
                elif mode == "full_body":
                    mode_value = 2
                else:  # mode == "custom"
                    mode_value = 3
                
                # 最终编码 (0-15)
                mode_code = gender_code * 4 + mode_value
                
                # 合成最终图像 -------------------------------------------------
                with Timer("合成最终图像"):
                    final_image, final_mask_L, final_mask_R, combined_mask = ImageProcessor.combine_images(
                        final_face_L, final_mask_L, final_face_R, final_mask_R, target_w, target_h, bg_color
                    )
                
                # 转换为输出格式
                image_out, mask_L_out, mask_R_out, mask_combined_out = self._wrap_outputs(
                    final_image, final_mask_L, final_mask_R, combined_mask
                )
                
                return (image_out, mask_L_out, mask_R_out, mask_combined_out, gender_code_L, gender_code_R, mode_code)
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 创建空白图像作为输出
            blank = np.zeros((512, 512, 3), dtype=np.float32)
            blank_tensor = torch.from_numpy(blank).unsqueeze(0)
            blank_mask = torch.zeros((512, 512), dtype=torch.float32).unsqueeze(0)
            return (blank_tensor, blank_mask, blank_mask, blank_mask, 0, 0, 0)  # 出错时默认返回女性组合肖像模式(0)
    
    def _wrap_outputs(self, image, mask_L, mask_R, mask_combined):
        """输出格式转换"""
        image_tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0)
        mask_L_tensor = torch.from_numpy(mask_L.astype(np.float32)/255.0).unsqueeze(0)
        mask_R_tensor = torch.from_numpy(mask_R.astype(np.float32)/255.0).unsqueeze(0)
        mask_combined_tensor = torch.from_numpy(mask_combined.astype(np.float32)/255.0).unsqueeze(0)
        return (image_tensor, mask_L_tensor, mask_R_tensor, mask_combined_tensor)