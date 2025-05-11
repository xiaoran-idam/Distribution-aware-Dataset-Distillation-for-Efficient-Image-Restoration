import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from diffusers import DiffusionPipeline  # 使用最新扩散模型库
from transformers import ViTModel
import numpy as np
from PIL import Image
import os
from torch.distributions import kl_divergence
import torch.nn.functional as F

# --------------------- 1. 数据预处理与加载 ---------------------
class ImagePairDataset(Dataset):
    """加载LQ-HQ图像对"""
    def __init__(self, lq_dir, hq_dir, transform=None):
        self.lq_paths = sorted([os.path.join(lq_dir, f) for f in os.listdir(lq_dir)])
        self.hq_paths = sorted([os.path.join(hq_dir, f) for f in os.listdir(hq_dir)])
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),  # 下采样加速处理
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return min(len(self.lq_paths), len(self.hq_paths))
    
    def __getitem__(self, idx):
        lq_img = Image.open(self.lq_paths[idx]).convert("RGB")
        hq_img = Image.open(self.hq_paths[idx]).convert("RGB")
        return self.transform(lq_img), self.transform(hq_img)

# --------------------- 2. ViT复杂度计算与子集选择 ---------------------
def calculate_entropy(features):
    """计算特征图的熵值"""
    prob = F.softmax(features.view(-1), dim=0)
    return -torch.sum(prob * torch.log(prob + 1e-8))

def select_complex_subset(dataset, top_k=0.02):
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.eval()
    entropies = []
    
    with torch.no_grad():
        for lq, hq in DataLoader(dataset, batch_size=1):  # 使用HQ图像计算复杂度
            outputs = vit(hq)
            features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
            entropy = calculate_entropy(features)
            entropies.append(entropy.item())
    
    sorted_indices = np.argsort(entropies)[::-1]  # 降序排列
    subset_size = int(len(dataset) * top_k)
    return Subset(dataset, sorted_indices[:subset_size])

# --------------------- 3. 扩散模型生成合成数据 ---------------------
class SyntheticDataset(Dataset):
    """加载生成的合成数据"""
    def __init__(self, synth_dir, transform=None):
        self.img_paths = [os.path.join(synth_dir, f) for f in os.listdir(synth_dir)]
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)  # 仅LQ，需配对HQ

def generate_and_save_synthetic(num, save_dir, prompt="clean high-quality image"):
    os.makedirs(save_dir, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    
    for i in range(num):
        image = pipe(prompt=prompt).images[0]
        image.save(os.path.join(save_dir, f"synth_{i}.png"))

# --------------------- 4. 完整8层CNN结构 ---------------------
class AlignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, padding=1)  # 输出通道与输入一致
        )
    
    def forward(self, x):
        return self.net(x)

# --------------------- 5. 联合损失函数（L2 + KL）---------------------
def composite_loss(real_feats, synth_feats, alpha=0.5):
    # L2像素损失
    l2_loss = F.mse_loss(real_feats, synth_feats)
    
    # KL散度（特征分布）
    real_probs = F.softmax(real_feats.view(-1), dim=0)
    synth_probs = F.softmax(synth_feats.view(-1), dim=0)
    kl_loss = kl_divergence(
        torch.distributions.Categorical(real_probs),
        torch.distributions.Categorical(synth_probs)
    ).mean()
    
    return alpha * l2_loss + (1 - alpha) * kl_loss

# --------------------- 6. 训练流程 ---------------------
def train_tripled():
    # 配置参数
    data_root = "I-HAZY"
    synth_dir = "generated_synth"
    batch_size = 16
    grad_accum_steps = 4  # 梯度累积步数
    
    # 加载原始数据集
    full_dataset = ImagePairDataset(
        lq_dir=os.path.join(data_root, "LQ"),
        hq_dir=os.path.join(data_root, "HQ")
    )
    
    # 选择高复杂度子集（2%）
    subset = select_complex_subset(full_dataset, top_k=0.02)
    
    # 生成合成数据（示例生成10张）
    generate_and_save_synthetic(num=10, save_dir=synth_dir)
    synth_dataset = SyntheticDataset(synth_dir, transform=subset.dataset.transform)
    
    # 合并数据集
    train_dataset = ConcatDataset([subset, synth_dataset])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    cnn = AlignmentCNN()
    optimizer = optim.AdamW(cnn.parameters(), lr=3e-4)
    
    # 训练循环
    cnn.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            lq_imgs = batch[0]  # 假设batch包含LQ图像
            
            # 前向传播
            adjusted_imgs = cnn(lq_imgs)
            
            # 计算联合损失（需获取真实HQ特征）
            with torch.no_grad():
                real_feats = vit.extract_features(lq_imgs)  # 需替换实际特征提取
            
            loss = composite_loss(real_feats, adjusted_imgs)
            
            # 梯度累积
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 保存模型
        torch.save(cnn.state_dict(), f"alignment_cnn_epoch{epoch}.pth")

if __name__ == "__main__":
    train_tripled()