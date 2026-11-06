import os
import random
from PIL import Image

# 1. 过滤：只保留 flower_photos 下的文件夹（排除 LICENSE.txt 等文件）
base_dir = 'flower_photos'
file_name = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# 如果 base_dir 不存在，创建它（注意：你的原代码是先 listdir 再判断 exists，顺序反了，如果不存在会先报错）
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

s = ['train', 'val']
for j in s:
    for i in file_name:
        # 2. 加上 exist_ok=True，防止重复运行时报错
        os.makedirs(os.path.join(base_dir, j, i), exist_ok=True)

for i in file_name:
    print(f"正在处理类别: {i}")
    class_dir = os.path.join(base_dir, i)
    file = os.listdir(class_dir)
    print(f"该类别的图片数量: {len(file)}")

    # 3. 防止某些类别的图片不足100张导致 random.sample 报错
    val_num = min(100, len(file))
    num_val = random.sample(file, val_num)

    for j in file:
        if j in num_val:
            # 建议使用 os.path.join 拼接路径，跨平台更安全（Windows/Linux/Mac通用）
            img_path = os.path.join(class_dir, j)
            save_path = os.path.join(base_dir, 'val', i, j)
        else:
            img_path = os.path.join(class_dir, j)
            save_path = os.path.join(base_dir, 'train', i, j)

        # 读取并保存图片
        try:
            img = Image.open(img_path)
            img.save(save_path)
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")

print("数据划分完成！")