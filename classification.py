import torch
import torch.nn as nn
import torch.optim as optim
from graph_construting import *
from markov_chain import *
from sklearn.metrics import classification_report

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
        return self.model(x)

def train(X_train,y_train,num_epochs,model,optimizer):
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

def test(X_test,y_test,model):
    # 测试模型
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).argmax(axis=1)
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f'Test Accuracy: {accuracy:.4f}')
        print("\nClassification Report:\n", classification_report(y_test, predictions.numpy(),target_names=['Healthy', 'Sick'], labels=[0, 1]))


def split_dataset(X, y, train_ratio=0.8, test_ratio=0.2, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    train_indices = indices[:train_end]
    test_indices = indices[train_end:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

if __name__ == "__main__":

    #eeg_data,lables,corr_matrixs = read_all_data()
    #print(len(eeg_data),len(eeg_data[0]),eeg_data[0][0],len(lables),len(corr_matrixs),len(corr_matrixs[0]),corr_matrixs[0][0])
    #X,y = get_features(eeg_data,lables,corr_matrixs)
    #X为 (sample_num, feather_num(24nodes+1),channal_num,channal_num); y为(sample_num,)
    #np.save("X1.npy", X)
    #np.save("y1.npy", y)

    X = np.load("X1.npy",allow_pickle=True)
    y = np.load("y1.npy",allow_pickle=True)
    print(X.shape,y.shape)#(1158, 25, 24, 24) (1158,)
    print(type(X))#<class 'numpy.ndarray'>

    X_train, y_train, X_test, y_test = split_dataset(X,y)
    print(y_test)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 超参数设置
    input_dim = X_train_flat.shape[1]  # 输入特征维度
    hidden_dim = 128  # 隐藏层单元数
    output_dim = 2  # 输出类别数
    learning_rate = 0.01
    num_epochs = 50
    # 初始化模型、损失函数和优化器
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #X_train_flat = np.array(X_train_flat, dtype=np.float32)
    print(X_train_flat.shape)
    train(X_train_flat,y_train,num_epochs,model,optimizer)
    test(X_test_flat,y_test,model)