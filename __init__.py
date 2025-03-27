# 使用更安全的导入方式，避免循环依赖
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 逐个导入模块，出错时提供详细日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUI_FaceAlignPaste")

try:
    from .face_align_node import FaceAlignDouble
    NODE_CLASS_MAPPINGS["FaceAlignDouble"] = FaceAlignDouble
    NODE_DISPLAY_NAME_MAPPINGS["FaceAlignDouble"] = "Face Align Double @ CHAOS"
    logger.info("成功加载 FaceAlignDouble 节点")
except Exception as e:
    logger.error(f"加载 FaceAlignDouble 节点失败: {str(e)}")

try:
    from .face_align_single import FaceAlignSingle
    NODE_CLASS_MAPPINGS["FaceAlignSingle"] = FaceAlignSingle
    NODE_DISPLAY_NAME_MAPPINGS["FaceAlignSingle"] = "Face Align Single @ CHAOS"
    logger.info("成功加载 FaceAlignSingle 节点")
except Exception as e:
    logger.error(f"加载 FaceAlignSingle 节点失败: {str(e)}")

try:
    from .face_autofit_single import FaceAutoFitSingle
    NODE_CLASS_MAPPINGS["FaceAutoFitSingle"] = FaceAutoFitSingle
    NODE_DISPLAY_NAME_MAPPINGS["FaceAutoFitSingle"] = "Face AutoFit Single @ CHAOS"
    logger.info("成功加载 FaceAutoFitSingle 节点")
except Exception as e:
    logger.error(f"加载 FaceAutoFitSingle 节点失败: {str(e)}")

try:
    from .face_autofit_double import FaceAutoFitDouble
    NODE_CLASS_MAPPINGS["FaceAutoFitDouble"] = FaceAutoFitDouble
    NODE_DISPLAY_NAME_MAPPINGS["FaceAutoFitDouble"] = "Face AutoFit Double @ CHAOS"
    logger.info("成功加载 FaceAutoFitDouble 节点")
except Exception as e:
    logger.error(f"加载 FaceAutoFitDouble 节点失败: {str(e)}")

try:
    from .rotate_node import ImageRotateCHAOS
    NODE_CLASS_MAPPINGS["ImageRotateCHAOS"] = ImageRotateCHAOS
    NODE_DISPLAY_NAME_MAPPINGS["ImageRotateCHAOS"] = "Image Rotate @ CHAOS"
    logger.info("成功加载 ImageRotateCHAOS 节点")
except Exception as e:
    logger.error(f"加载 ImageRotateCHAOS 节点失败: {str(e)}")

try:
    from .gender_recognition_node import GenderRecognitionNode
    NODE_CLASS_MAPPINGS["GenderRecognitionNode"] = GenderRecognitionNode
    NODE_DISPLAY_NAME_MAPPINGS["GenderRecognitionNode"] = "Gender Recognition"
    logger.info("成功加载 GenderRecognitionNode 节点")
except Exception as e:
    logger.error(f"加载 GenderRecognitionNode 节点失败: {str(e)}")

# 删除旧的导入语句，只保留新的导入语句
try:
    from .prompt_selector import PromptSelectorDoubleCHAOS, PromptSelectorSingleCHAOS
    NODE_CLASS_MAPPINGS["PromptSelectorDoubleCHAOS"] = PromptSelectorDoubleCHAOS
    NODE_DISPLAY_NAME_MAPPINGS["PromptSelectorDoubleCHAOS"] = "Prompt Selector Double @ CHAOS"
    NODE_CLASS_MAPPINGS["PromptSelectorSingleCHAOS"] = PromptSelectorSingleCHAOS
    NODE_DISPLAY_NAME_MAPPINGS["PromptSelectorSingleCHAOS"] = "Prompt Selector Single @ CHAOS"
    logger.info("成功加载 Prompt Selector 节点")
except Exception as e:
    logger.error(f"加载 Prompt Selector 节点失败: {str(e)}")

VERSION = "1.1.0"  # 更新版本号
