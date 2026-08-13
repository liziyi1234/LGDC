# cyclic_shift.py
import random
import numpy as np
from PIL import Image

class ERPHorizontalCyclicShift:
  
    def __init__(self, max_shift_ratio=0.3, p=0.5):
       
        self.max_shift_ratio = max_shift_ratio
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # 随机决定平移方向和距离（正数右移，负数左移）
        max_shift = int(img.width * self.max_shift_ratio)
        shift_pixels = random.randint(-max_shift, max_shift)
        
        # 转换为numpy数组并循环平移
        img_array = np.array(img)
        shifted_array = np.roll(img_array, shift=shift_pixels, axis=1)
        
        return Image.fromarray(shifted_array)
    
    def __repr__(self):
        return f"ERPHorizontalCyclicShift(max_shift_ratio={self.max_shift_ratio}, p={self.p})"