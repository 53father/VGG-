import torch
from PIL import Image
from torchvision import transforms
import json
import os
from model import vgg

# ==========================================
# 1. 设置设备与加载类别字典
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)
except Exception as e:
    print(f"❌ 找不到 class_indices.json，请检查它是否在同一个文件夹下。报错: {e}")
    exit()

# ==========================================
# 2. 加载模型结构并导入你的 92% 大脑
# ==========================================
model_name = "vgg16"
net = vgg(model_name=model_name, init_weights=False)

model_weights_path = './vgg.pth'
if not os.path.exists(model_weights_path):
    print(f"❌ 找不到模型文件 {model_weights_path}！")
    exit()

print("正在加载训练好的 AI 大脑...")
# map_location=device 确保即使你换了电脑也不会报错
net.load_state_dict(torch.load(model_weights_path, map_location=device))
net.to(device)
net.eval()  # 切换到预测模式

# ==========================================
# 3. 图片预处理（必须和训练时一模一样）
# ==========================================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==========================================
# 4. 批量读取 test 文件夹并预测
# ==========================================
test_dir = "test"

# 检查文件夹是否存在
if not os.path.exists(test_dir):
    print(f"❌ 找不到 '{test_dir}' 文件夹！请新建一个名为 test 的文件夹，并把图片放进去。")
    exit()

# 获取文件夹里的所有文件
all_files = os.listdir(test_dir)
# 过滤出图片文件（忽略掉可能存在的 .DS_Store 或 txt 等杂文件）
img_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if len(img_files) == 0:
    print(f"⚠️ 文件夹 '{test_dir}' 里没有找到任何图片，请放几张图片进去！")
    exit()

print(f"\n在 test 文件夹中找到了 {len(img_files)} 张图片，开始预测...\n")

# 循环处理每一张图片
for img_name in img_files:
    img_path = os.path.join(test_dir, img_name)

    try:
        # 读取图片，并强制转换为 RGB（防止有些 png 是带透明通道的四通道图，模型会报错）
        img = Image.open(img_path).convert('RGB')

        # 预处理图片，并增加 batch 维度
        img_tensor = data_transform(img).unsqueeze(dim=0).to(device)

        with torch.no_grad():
            outputs = net(img_tensor)
            predict = torch.softmax(outputs, dim=1)
            predict_cla = torch.argmax(predict).item()

        class_name = class_indict[str(predict_cla)]
        probability = predict[0][predict_cla].item()

        # 打印单张图片的结果
        print(f"图片: {img_name:<20} |  预测:【{class_name}】 | 确信度: {probability * 100:.2f}%")

    except Exception as e:
        print(f"❌ 处理图片 {img_name} 时出错，跳过。错误信息: {e}")

print("\n全部预测完成！")