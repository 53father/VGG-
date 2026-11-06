import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from model import vgg
import os
import json
import time

# ==========================================
# 1. 设置设备（CPU 或 GPU）
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")
if device.type == 'cpu':
    print("⚠️ 警告：你当前使用的是 CPU，训练 VGG16 会非常非常慢，请做好心理准备！")

# ==========================================
# 2. 数据预处理（不用改）
# ==========================================
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# ==========================================
# 3. 设置数据路径（🔥 重点！）
# ==========================================
# 获取当前 train.py 所在的文件夹路径（即 D:\python\VGG）
current_dir = os.getcwd()

# 你的数据集文件夹名字叫啥，这里就填啥。
# 根据你之前的操作，应该是 "flower_photos"。如果是 "flower_data"，请把下面这行的 flower_photos 改掉。
data_folder_name = "flower_photos"
image_path = os.path.join(current_dir, data_folder_name)

# 检查一下路径到底对不对，防止小白找不到路
print(f"正在寻找数据集，路径为：{image_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"❌ 找不到数据集文件夹！请检查你的电脑里有没有叫 '{data_folder_name}' 的文件夹，并且里面要有 train 和 val 文件夹！")

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)
print(f"成功加载训练集，共有 {train_num} 张图片。")

# 将类别映射保存为 json 文件（预测的时候要用到）
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# ==========================================
# 4. 加载验证集并创建 DataLoader
# ==========================================
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
print(f"成功加载验证集，共有 {val_num} 张图片。")

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,  # 验证集不需要打乱
                                              num_workers=0)

# ==========================================
# 5. 初始化模型并加载“学霸笔记”
# ==========================================
model_name = "vgg16"
net = vgg(model_name=model_name, init_weights=True)

# 加载预训练权重 vgg16.pth
model_weights_path = './vgg16.pth'
if os.path.exists(model_weights_path):
    print("正在加载 vgg16.pth 预训练权重...")
    try:
        # strict=False 表示允许最后一层分类层不匹配，自动跳过
        net.load_state_dict(torch.load(model_weights_path), strict=False)
        print("✅ 预训练权重加载成功！")
    except Exception as e:
        print(f"⚠️ 权重加载失败，原因是: {e}")
else:
    print("⚠️ 没有找到 vgg16.pth 文件！将使用完全空白的模型从零开始训练（效果可能很差）。")
    print("👉 建议去网上下载 vgg16.pth 放在 train.py 同级目录下。")

net.to(device)

# ==========================================
# 6. 定义损失函数和优化器
# ==========================================
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

# ==========================================
# 7. 开始训练
# ==========================================
if __name__ == '__main__':
    save_path = './vgg.pth'
    best_acc = 0.0
    epochs = 20  # 训练总轮数

    for epoch in range(epochs):
        # --- 训练阶段 ---
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()

        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\r训练进度: {:^3.0f}%[{}->{}] 当前Loss: {:.3f}".format(int(rate * 100), a, b, loss), end="")

        print()  # 换行
        print(f"本轮训练耗时: {time.perf_counter() - t1:.2f} 秒")

        # --- 验证阶段 ---
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()

            val_accurate = acc / val_num

            # 保存最好的模型
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

            print('[第 %d 轮] 平均训练Loss: %.3f  验证集准确率: %.3f  历史最高准确率: %.3f' %
                  (epoch + 1, running_loss / len(train_loader), val_accurate, best_acc))

    print('🎉 训练全部完成！最好的模型已保存为 vgg.pth')