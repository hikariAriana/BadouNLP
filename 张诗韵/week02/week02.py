"""

2024/11/30:
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# 生成数据与标签
def generate_data(batch_size):
    # 生成五维随机向量
    data = np.random.rand(batch_size, 5)

    # 每个样本的标签是最大值所在的维度
    labels = np.argmax(data, axis=1)

    # 转换为PyTorch张量
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return data, labels


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x)


# 超参数设置
batch_size = 30
epochs = 200
learning_rate = 0.01

# 创建模型
model = SimpleNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    data, labels = generate_data(batch_size)

    # 将数据和标签转移到GPU（如果有的话）
    data, labels = data, labels

    # 前向传播
    outputs = model(data)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")


# 测试模型
def test_model(model, batch_size=10):
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        data, labels = generate_data(batch_size)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)  # 预测类别

        accuracy = (predicted == labels).sum().item() / batch_size
        print(f'Accuracy: {accuracy:.4f}')


test_model(model)
