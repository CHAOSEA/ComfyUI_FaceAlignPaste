import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceAutoFit")

# 性能计时器
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