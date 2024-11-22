import torch
import torch.nn as nn
import torch.optim as optim
from graph_construting import read_data

# 定义 MLP 模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)  # 用于多分类
        )

    def forward(self, x):
        x = np.reshape(x,)
        return self.model(x)

# 超参数设置
input_dim = X_train.shape[1]  # 输入特征维度
hidden_dim = 128  # 隐藏层单元数
output_dim = len(np.unique(y))  # 输出类别数
learning_rate = 0.01
num_epochs = 50
# 初始化模型、损失函数和优化器
model = MLPClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(X_train,y_train):
    # 转换数据为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def test(X_test,y_test):
    # 测试模型
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).argmax(axis=1)
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f'Test Accuracy: {accuracy:.4f}')
        print("\nClassification Report:\n", classification_report(y_test, predictions.numpy()))

if __name__ == "__main__":
    eeg_data,lables,corr_matrixs = read_data()
    # 展平时间和通道维度
    eeg_data = np.array(eeg_data)
    train_ratio = 0.8
    num_train = int(num_samples * train_ratio)
    X_train, X_test = X_flat[:num_train], X_flat[num_train:]
    X_train_flat = eeg_data.reshape(eeg_data.shape[0], -1)  # 形状：(样本数, 时间×通道)
    train(eeg_data,lables)