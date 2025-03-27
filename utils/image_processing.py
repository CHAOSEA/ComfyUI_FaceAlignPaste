import cv2
import numpy as np
from typing import Tuple, Dict
from .logging_utils import logger

class ImageProcessor:
    @staticmethod
    def smart_scaling(image, mask, target_h, head_ratio, face_size, face_center, face_info) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """智能缩放算法 - 根据人脸比例调整，保持人脸中心不变"""
        # 移除全局缩放系数，使用用户提供的face_size参数
        global_scale_factor = 1.0  # 默认不额外缩放
        
        # 应用用户指定的缩放系数
        adjusted_face_size = face_size * global_scale_factor
        logger.info(f"调整人脸大小系数为 {adjusted_face_size:.2f}")
        
        # 优先使用检测到的人脸区域进行缩放
        if face_info["face_height"] is not None:
            try:
                # 使用人脸高度计算缩放比例
                face_height = face_info["face_height"]
                target_face_height = target_h * head_ratio
                scale_factor = (target_face_height / face_height) * adjusted_face_size
                
                # 强制限制缩放比例在合理范围内
                min_scale = 0.1
                max_scale = 2.0
                scale_factor = max(min(scale_factor, max_scale), min_scale)
                
                # 记录缩放比例和目标人脸高度，用于后续处理
                face_info["scale_factor"] = scale_factor
                face_info["target_face_height"] = target_face_height
                
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
                scale_factor = (target_face_height / estimated_face_height) * adjusted_face_size
                
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
            scale_factor = (target_h * 0.7) / image.shape[0] * adjusted_face_size
            
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
    
    @staticmethod
    def rotate_image(image, mask, angle, bg_color, center) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """改进的旋转函数 - 支持扩展画布以避免裁剪"""
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
    
    @staticmethod
    def apply_positioning(image, mask, target_w, target_h, bg_color, move_x, move_y, 
                         face_center, eye_position_ratio, gender, position="center"):
        """精确定位算法 - 根据人脸中心和眼睛位置比例放置人物"""
        # 创建目标画布
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * bg_color
        canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # 计算眼睛应该在画布中的垂直位置
        target_eye_y = int(target_h * eye_position_ratio)
        
        # 计算水平位置 - 调整左右人脸的间距
        if position == "left":
            # 左侧人物 - 放在画布左侧约35%处
            target_eye_x = int(target_w * 0.35)
        elif position == "right":
            # 右侧人物 - 放在画布右侧约70%处
            target_eye_x = int(target_w * 0.70)
        else:
            # 居中
            target_eye_x = target_w // 2
        
        # 修正移动逻辑：X为正向右移动，Y为正向上移动
        target_eye_x += int(move_x)  # 向右移动
        target_eye_y -= int(move_y)  # 向上移动（注意这里是减法，因为图像坐标系Y轴向下）
        
        # 计算图像应该放置的位置
        face_x, face_y = face_center
        x_offset = target_eye_x - face_x
        y_offset = target_eye_y - face_y
        
        # 计算图像在画布上的位置
        x1 = x_offset
        y1 = y_offset
        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]
        
        # 计算有效的粘贴区域（处理图像超出画布的情况）
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = min(image.shape[1], target_w - x1)
        src_y2 = min(image.shape[0], target_h - y1)
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(target_w, x2)
        dst_y2 = min(target_h, y2)
        
        # 检查是否有有效的粘贴区域
        if src_x1 < src_x2 and src_y1 < src_y2 and dst_x1 < dst_x2 and dst_y1 < dst_y2:
            # 粘贴图像到画布
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
            canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
        
        # 计算最终的人脸中心点
        final_face_center = (target_eye_x, target_eye_y)
        
        return canvas, canvas_mask, final_face_center
    
    @staticmethod
    def combine_images(image_L, mask_L, image_R, mask_R, target_w, target_h, bg_color):
        """合成最终图像 - 将左右两个人物合成到一个画布上"""
        # 创建最终画布
        final_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * bg_color
        combined_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # 先绘制左侧人物（后层）
        mask_L_bool = mask_L > 0
        final_image[mask_L_bool] = image_L[mask_L_bool]
        combined_mask[mask_L_bool] = 255
        
        # 再绘制右侧人物（前层）
        mask_R_bool = mask_R > 0
        final_image[mask_R_bool] = image_R[mask_R_bool]
        combined_mask[mask_R_bool] = 255
        
        # 返回最终图像和各个遮罩
        return final_image, mask_L, mask_R, combined_mask
    
    @staticmethod
    def check_boundary(mask, name):
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
    
    @staticmethod
    def get_background_color(bg_type) -> Tuple[int, int, int]:
        """背景颜色管理系统"""
        return (255, 255, 255) if bg_type == "white" else (192, 192, 192)
    
    @staticmethod
    def prepare_inputs(image_tensor, mask_tensor) -> Tuple[np.ndarray, np.ndarray]:
        """输入预处理"""
        image = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)[:, :, :3]
        mask = (mask_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        return image, mask
    
    @staticmethod
    def wrap_outputs(image, mask_L=None, mask_R=None, mask_combined=None) -> Tuple:
        """输出格式转换"""
        image_tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0)
        
        if mask_L is not None and mask_R is not None and mask_combined is not None:
            # 双人模式输出
            mask_L_tensor = torch.from_numpy(mask_L.astype(np.float32)/255.0).unsqueeze(0)
            mask_R_tensor = torch.from_numpy(mask_R.astype(np.float32)/255.0).unsqueeze(0)
            mask_combined_tensor = torch.from_numpy(mask_combined.astype(np.float32)/255.0).unsqueeze(0)
            return (image_tensor, mask_L_tensor, mask_R_tensor, mask_combined_tensor)
        else:
            # 单人模式输出
            mask_tensor = torch.from_numpy((mask_L if mask_L is not None else mask_combined).astype(np.float32)/255.0).unsqueeze(0)
            return (image_tensor, mask_tensor)