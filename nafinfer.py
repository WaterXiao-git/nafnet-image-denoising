import torch
import os
import cv2
import numpy as np
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet
from torch.nn import functional as F
from glob import glob
from tqdm import tqdm

# ------------ 参数配置 ------------ #
test_img_dir = 'datasets/val'  # 有噪声图像的文件夹路径
save_dir = 'datasets/clean' # 输出文件夹
model_path = 'NAFNet-SIDD-width32.pth'  # 模型路径
os.makedirs(save_dir, exist_ok=True)

# ------------ 加载模型 ------------ #
model = NAFNet(width=32, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
net_data = torch.load(model_path, map_location='cpu')
model.load_state_dict(net_data['params'], strict=True)
model.eval().cuda()

# ------------ 推理函数 ------------ #
def denoise_image(img_path):
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # 确保输入尺寸为32的倍数（NAFNet要求）
    new_h = (h + 31) // 32 * 32
    new_w = (w + 31) // 32 * 32
    pad_img = cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT)

    # 转换为Tensor
    img_tensor = torch.from_numpy(pad_img.transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(img_tensor)
        out = out[..., :h, :w]  # 去掉 padding 部分
        out_img = out.squeeze().cpu().numpy().transpose(1, 2, 0)
        out_img = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
        return out_img

# ------------ 批量处理测试图像 ------------ #
img_paths = glob(os.path.join(test_img_dir, '*'))
for img_path in tqdm(img_paths, desc='Processing'):
    out_img = denoise_image(img_path)
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, filename), out_img)

print(f'\n✅ 所有图像已去噪完成，保存在 {save_dir}')
