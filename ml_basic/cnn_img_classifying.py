from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

"""
最小化版的 CIFAR-10 图像分类训练 + 推理脚本，具体流程：
    训练：用一个小型 CNN 在 CIFAR-10 数据集上训练
    保存：在验证集精度最高时保存模型权重
    预测：加载训练好的模型，对任意图片做分类
"""

# =============== 可配参数 =================
DATA_DIR = Path("/Users/yuanjie05/Downloads")  # CIFAR-10 缓存/下载目录
SAVE_DIR = Path("./runs/cifar10_min_cnn")  # 模型与日志保存目录
NUM_EPOCHS = 5  # 训练轮数（示例用小轮数数）
BATCH_SIZE = 128  # 每次喂入的图片数，与显存/内存相关；大批次更稳定但占显存
LEARNING_RATE = 1e-3  # 学习率
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9  # 用在SGD时；如果用Adam可忽略
OPTIMIZER = "adam"  # "adam"或"sgd"
USE_AUG = True  # 是否启用数据增强（随机裁剪/翻转）
NUM_WORKERS = 2
SEED = 42
CKPT_NAME = "best_cifar10_cnn.pth"  # 统一管理权重文件名
SKIP_TRAIN_IF_CKPT_EXISTS = True  # 有权重且提供IMG_PATH时，直接跳过训练去预测

# 指定一张要预测的图片（任意来源的RGB图片），空字符串则跳过预测
IMG_PATH = "/Users/yuanjie05/Downloads/tiger.jpg"


# ========================================

# 固定随机种子（结果更可复现）
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 简单的CNN（32x32输入）
class SmallCNN(nn.Module):
    """
    构造一个标准的小型卷积神经网络（CNN），输入是 32×32×3 的 CIFAR-10 图像：
    - 两层卷积（3→32→64） + 池化 → 下采样到 16×16
    - 再两层卷积（64→128→128） + 池化 → 下采样到 8×8
    - flatten（拉平）→ 两层全连接
    - 输出 10 个类别
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 → 32 提取低级边缘特征
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 → 64 提取更复杂局部特征
        self.pool = nn.MaxPool2d(2, 2)  # 池化层降采样一半 (32→16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 → 128 继续提取高级特征（形状、结构）
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 保持尺寸不变，但提高语义复杂度
        self.pool2 = nn.MaxPool2d(2, 2)  # 再次降采样 (16→8)
        self.dropout = nn.Dropout(0.3)  # 训练时随机丢掉 30% 的神经元，防止过拟合
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 压缩信息
        self.fc2 = nn.Linear(256, num_classes)  # 输出 10 类 logits

    def forward(self, x):
        x = F.relu(self.conv1(x))  # relu 把所有负数清零，只保留正数
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0),
                   -1)  # 相当于torch.flatten(x, start_dim=1)，把卷积层输出的三维特征 [batch_size, 通道数, 高, 宽]，变成二维 [batch_size, 通道数 × 高 × 宽]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_transforms(train=True):
    """
    训练集随机裁剪/水平翻转；所有数据按 CIFAR-10 统计量做归一化
    - 增强提升泛化：随机视角/位移让模型更鲁棒。padding=4 再裁剪模拟轻微平移。
    - Normalize把像素分布对齐，加速收敛、稳定训练。均值/方差是CIFAR-10 官方统计
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if train and USE_AUG:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def prepare_data(device):
    assert (DATA_DIR / "cifar-10-batches-py").exists(), \
        f"找不到本地数据目录：{DATA_DIR / 'cifar-10-batches-py'}"

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=False, transform=get_transforms(train=True)
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=False, transform=get_transforms(train=False)
    )

    pin = (device.type == "cuda")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)
    classes = train_set.classes
    return train_loader, test_loader, classes


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return running_loss / n, running_acc / n


def get_optimizer(model):
    if OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                               momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)


def predict_image(model, img_path, classes, device):
    """
    对任意图片做分类：会自动RGB+resize(32,32)+normalize
    让推理时的数据形态与训练时严格对齐（尺寸、通道、归一化），在评估模式下做一次前向，把 logits 通过 softmax 变成概率，再取 Top-K。
    只要训练ok，就能稳定、可复现地给出合理的类别判断
    """
    # 这是 CIFAR-10 官方统计的 RGB 通道均值/方差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    # 推理阶段要用与训练阶段完全一致的归一化，否则模型“看到”的像素分布不同，表现会明显变差
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()  # 切换到推理态：关闭 Dropout 的随机丢弃，使用滑动均值/方差而不是 batch 统计，保证推理稳定可复现
    with torch.no_grad():  # 禁用 autograd，节省显存和算力（推理不需要反传）
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
    topk = probs.argsort()[::-1][:3]
    print(f"\n[Predict] {img_path}")
    for i, k in enumerate(topk, 1):
        print(f"Top-{i}: {classes[k]}  prob={probs[k]:.4f}")
    return classes[topk[0]]


def main():
    ckpt_path = SAVE_DIR / CKPT_NAME

    if SKIP_TRAIN_IF_CKPT_EXISTS and not IMG_PATH and ckpt_path.exists():
        print(f"[Skip] 已存在训练好的模型：{ckpt_path}，未指定图片，因此直接退出。")
        return

    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"[Device] {device}")

    # 如果：配置允许跳过训练+提供了图片路径+ckpt已存在->直接加载预测并 return
    if SKIP_TRAIN_IF_CKPT_EXISTS and IMG_PATH and Path(IMG_PATH).exists() and ckpt_path.exists():
        print(f"[FastPredict] 发现已训练权重：{ckpt_path}，直接加载进行预测。")
        ckpt = torch.load(ckpt_path, map_location=device)
        classes = ckpt["classes"]
        model = SmallCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(ckpt["model_state"])
        predict_image(model, IMG_PATH, classes, device)
        return

    train_loader, test_loader, classes = prepare_data(device)
    model = SmallCNN(num_classes=len(classes)).to(device)
    optimizer = get_optimizer(model)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f} acc={te_acc:.4f}")

        # 保存最好模型
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": {
                    "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "optimizer": OPTIMIZER,
                    "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE
                }
            }, SAVE_DIR / CKPT_NAME)
            print(f"[Save] 更新最佳模型 acc={best_acc:.4f} -> {SAVE_DIR / CKPT_NAME}")

    # 可选：对任意图片做一次预测
    if IMG_PATH and Path(IMG_PATH).exists():
        ckpt = torch.load(SAVE_DIR / CKPT_NAME, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        predict_image(model, IMG_PATH, classes, device)
    else:
        if IMG_PATH:
            print(f"[Warn] 指定的图片不存在：{IMG_PATH}")


if __name__ == "__main__":
    main()
