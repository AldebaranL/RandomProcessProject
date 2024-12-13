import torch
import torch.nn as nn
import torch.optim as optim
from graph_construting import *
from markov_chain import *
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import graphviz

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
            #nn.Sigmoid()  # 用于2分类
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
    return model,optimizer

def test(X_test,y_test,model):
    # 测试模型
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).argmax(axis=1)
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f'Test Accuracy: {accuracy:.4f}')
        #print("\nClassification Report:\n", classification_report(y_test, predictions.numpy(),target_names=['Healthy', 'Sick'], labels=[0, 1]))

    return accuracy,predictions.numpy()

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
    #X,y = get_MC_features(eeg_data,lables,corr_matrixs)
    #X为 (sample_num, feather_num(24nodes+1),channal_num,channal_num); y为(sample_num,)
    #np.save("X_MC_graph.npy", X)
    #np.save("y_no_cut.npy", y)

    X_MC1 = np.load("X_MC_degree_no_cut.npy",allow_pickle=True)
    X_MC2 = np.load("X_MC_graph_no_cut.npy", allow_pickle=True)
    y = np.load("y_no_cut.npy",allow_pickle=True)
    print(X_MC1.shape,X_MC2.shape,y.shape)#(1158, 25, 24, 24) (1158,)
    #print(type(X_MC))#<class 'numpy.ndarray'>
    X_MC_flat1 = X_MC1[:,:24,:,:].reshape(X_MC1.shape[0], -1)
    X_MC_flat2 = X_MC2[:,1,:,:].reshape(X_MC2.shape[0], -1)
    #X_wavelet = get_transform_feature(eeg_data)
    #np.save("X_wavelet.npy", X_wavelet)
    X_wavelet = np.load("X_wavelet_no_cut.npy", allow_pickle=True)
    print(X_wavelet.shape)
    X = np.hstack((X_MC_flat1,X_MC_flat2))
    #X = X_wavelet
    #X_train, y_train, X_test, y_test = split_dataset(X,y)
    print(X.shape)
    #y_train = np.expand_dims(y_train, 1)
    #y_test = np.expand_dims(y_test, 1)
    # 超参数设置
    input_dim = X.shape[1]  # 输入特征维度
    hidden_dim = 128  # 隐藏层单元数
    output_dim = 2  # 输出类别数
    learning_rate = 0.01
    num_epochs = 50
    print(input_dim)
    # n-CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracys,predictions = [],[]
    # 遍历每一折
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        # 初始化模型、损失函数和优化器
        model = MLPClassifier(input_dim, hidden_dim, output_dim)
        #criterion = nn.BCELoss()#
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]
        #X_train_flat = np.array(X_train_flat, dtype=np.float32)
        print(X_train.shape)
        train(X_train,y_train,num_epochs,model,optimizer)
        accuracy,prediction = test(X_test,y_test,model)
        accuracys.append(accuracy)
    print(np.mean(accuracys))