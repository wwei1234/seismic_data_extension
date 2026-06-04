import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import SeismicBandwidthDataset
from loss_function import SeismicBandwidthLoss
from U_Net_CBAM import UNet

def create_datasets(narrow_path, wide_path, patch_size=128, noise_levels=None, stride=64):
    """
    创建训练集、验证集和测试集
    
    数据划分策略:
    - 测试集: 时间轴1500-2000采样点
    - 其余数据: 80%训练集, 20%验证集
    """
    # 加载完整数据以确定维度
    narrow_full = np.load(narrow_path)
    total_time_samples = narrow_full.shape[0]
    
    print(f"总时间采样点: {total_time_samples}")
    print(f"数据划分:")
    
    # 测试集: 1500-2000
    test_start, test_end = 1500, 2000
    print(f"  测试集: [{test_start}, {test_end})")
    
    # 其余数据: 0-1500 和 2000-end
    remaining_ranges = [(350, test_start), (test_end, total_time_samples)]
    
    # 计算训练集和验证集的分割点
    # 使用 0-1500 的80% 作为训练集
    train_end = 350 + int((test_start - 350) * 0.8)
    print(f"  训练集: [350, {train_end}) 和 [{test_end}, {total_time_samples})")
    print(f"  验证集: [{train_end}, {test_start})")
    
    # 创建三个数据集
    # 训练集 - 使用翻转和多噪声增强
    train_dataset_1 = SeismicBandwidthDataset(
        narrow_path, wide_path,
        patch_size=patch_size,
        noise_levels=noise_levels if noise_levels else [0.10, 0.15, 0.20],
        use_flip=True,
        stride=stride,
        time_range=(350, train_end)
    )
    
    train_dataset_2 = SeismicBandwidthDataset(
        narrow_path, wide_path,
        patch_size=patch_size,
        noise_levels=noise_levels if noise_levels else [0.10, 0.15, 0.20],
        use_flip=True,
        stride=stride,
        time_range=(test_end, total_time_samples)
    )
    
    # 合并两个训练集
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
    
    # 验证集 - 不使用翻转，单一噪声
    val_dataset = SeismicBandwidthDataset(
        narrow_path, wide_path,
        patch_size=patch_size,
        noise_levels=[0.15],  # 验证集使用固定噪声
        use_flip=False,
        stride=stride,
        time_range=(train_end, test_start)
    )
    
    # 测试集 - 不使用翻转，单一噪声
    test_dataset = SeismicBandwidthDataset(
        narrow_path, wide_path,
        patch_size=patch_size,
        noise_levels=[0.15],  # 测试集使用固定噪声
        use_flip=False,
        stride=stride,
        time_range=(test_start, test_end)
    )
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_dataset, val_dataset, test_dataset


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_mae_loss = 0.0
    total_freq_loss = 0.0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累积损失
        total_loss += loss_dict['total']
        total_mae_loss += loss_dict['mae']
        total_freq_loss += loss_dict['freq']
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{loss_dict['total']:.4f}",
            'MAE': f"{loss_dict['mae']:.4f}",
            'Freq': f"{loss_dict['freq']:.4f}"
        })
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae_loss / len(dataloader)
    avg_freq = total_freq_loss / len(dataloader)
    
    return avg_loss, avg_mae, avg_freq


def validate(model, dataloader, criterion, device, epoch):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    total_mae_loss = 0.0
    total_freq_loss = 0.0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets)
            
            # 累积损失
            total_loss += loss_dict['total']
            total_mae_loss += loss_dict['mae']
            total_freq_loss += loss_dict['freq']
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total']:.4f}",
                'MAE': f"{loss_dict['mae']:.4f}",
                'Freq': f"{loss_dict['freq']:.4f}"
            })
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae_loss / len(dataloader)
    avg_freq = total_freq_loss / len(dataloader)
    
    return avg_loss, avg_mae, avg_freq


def train(model, train_loader, val_loader, criterion, optimizer, 
          num_epochs, device, save_dir='checkpoints', log_dir='logs', save_interval=10):
    """
    完整训练流程
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备 (cuda/cpu)
        save_dir: 模型保存目录
        log_dir: 日志保存目录
        save_interval: 保存间隔（每多少轮保存一次）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_mae': [],
        'train_freq': [],
        'val_loss': [],
        'val_mae': [],
        'val_freq': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    print(f"总轮数: {num_epochs}")
    print(f"设备: {device}")
    print(f"模型保存目录: {save_dir}")
    print(f"日志保存目录: {log_dir}")
    print(f"保存间隔: 每 {save_interval} 轮")
    print("="*70 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_mae, train_freq = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_mae, val_freq = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_freq'].append(train_freq)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_freq'].append(val_freq)
        
        # 打印epoch总结
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, Freq: {train_freq:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Freq: {val_freq:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 定期保存模型
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
            print(f"  ✓ 保存检查点: model_epoch_{epoch}.pth")
        
        print("-" * 70)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 保存训练历史到checkpoints目录（保持兼容性）
    np.save(os.path.join(save_dir, 'training_history.npy'), history)
    
    # 分别保存训练损失和验证损失到logs目录
    # 保存所有训练损失
    train_losses = {
        'total_loss': np.array(history['train_loss']),
        'mae_loss': np.array(history['train_mae']),
        'freq_loss': np.array(history['train_freq'])
    }
    np.save(os.path.join(log_dir, 'train_losses.npy'), train_losses)
    print(f"训练损失已保存: {os.path.join(log_dir, 'train_losses.npy')}")
    
    # 保存所有验证损失
    val_losses = {
        'total_loss': np.array(history['val_loss']),
        'mae_loss': np.array(history['val_mae']),
        'freq_loss': np.array(history['val_freq'])
    }
    np.save(os.path.join(log_dir, 'val_losses.npy'), val_losses)
    print(f"验证损失已保存: {os.path.join(log_dir, 'val_losses.npy')}")
    
    # 另外保存每个epoch的详细信息（可选，用于后续分析）
    epoch_info = {
        'epochs': np.arange(1, num_epochs + 1),
        'train_total': np.array(history['train_loss']),
        'train_mae': np.array(history['train_mae']),
        'train_freq': np.array(history['train_freq']),
        'val_total': np.array(history['val_loss']),
        'val_mae': np.array(history['val_mae']),
        'val_freq': np.array(history['val_freq']),
        'best_val_loss': best_val_loss
    }
    np.save(os.path.join(log_dir, 'epoch_info.npy'), epoch_info)
    print(f"Epoch详细信息已保存: {os.path.join(log_dir, 'epoch_info.npy')}")
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir)
    
    # 同时保存到logs目录
    plot_training_curves(history, log_dir)
    
    return history


def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 总损失
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_title('Total Loss Curve', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # MAE损失
    axes[1].plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE Loss', fontsize=14, fontweight='bold')
    axes[1].set_title('MAE Loss Curve', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 频域损失
    axes[2].plot(epochs, history['train_freq'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_freq'], 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Frequency Loss', fontsize=14, fontweight='bold')
    axes[2].set_title('Frequency Loss Curve', fontsize=16, fontweight='bold')
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")


# ==================== 主训练脚本 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    narrow_path = r'seismic_records\seismic_narrow_band.npy'
    wide_path = r'seismic_records\seismic_wide_band.npy'
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(
        narrow_path, wide_path,
        patch_size=128,
        noise_levels=[0.10, 0.15, 0.20],
        stride=64
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model = UNet().to(device)
    
    # 创建损失函数
    criterion = SeismicBandwidthLoss(lambda_mae=1.0, lambda_freq=0.5)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练模型
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        save_dir='checkpoints',
        save_interval=10
    )
    
    print("\n所有训练任务完成！")