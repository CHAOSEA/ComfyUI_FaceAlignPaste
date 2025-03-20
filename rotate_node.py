import numpy as np
from PIL import Image
import torch

class ImageRotateCHAOS:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 0.1
                }),
                # 修改背景选项为灰色
                "background": (["white", "black", "gray"], {"default": "white"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_image"
    CATEGORY = "image/transform"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (False,)

    def rotate_image(self, image, angle, background):
        try:
            img_np = image[0].cpu().numpy() * 255
            img_np = img_np.astype(np.uint8)
            
            # 统一转换为RGB模式
            pil_image = Image.fromarray(img_np).convert('RGB')
            
            # 设置填充颜色
            fill_color = {
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "gray": (128, 128, 128)  # 新增灰色背景
            }[background]

            # 计算旋转后的新尺寸
            w, h = pil_image.size
            radians = np.deg2rad(abs(angle))
            new_w = int(w * abs(np.cos(radians)) + h * abs(np.sin(radians)))
            new_h = int(w * abs(np.sin(radians)) + h * abs(np.cos(radians)))
            
            # 创建旋转后的画布
            # 统一使用RGB模式
            rotated_image = Image.new('RGB', (new_w, new_h), fill_color)
            
            # 执行旋转（移除透明通道处理）
            rotated_image.paste(pil_image, (
                (new_w - w) // 2,
                (new_h - h) // 2
            ))
            rotated_image = rotated_image.rotate(-angle, center=(new_w//2, new_h//2), 
                                                resample=Image.BILINEAR, fillcolor=fill_color)

            # 转换回tensor（保持3通道）
            rotated_np = np.array(rotated_image).astype(np.float32) / 255.0
            
            return (torch.from_numpy(rotated_np).unsqueeze(0),)
            
        except Exception as e:
            print(f"旋转处理失败: {str(e)}")
            return (image,)