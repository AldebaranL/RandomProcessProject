# 2024随机过程期末作业
林语盈 王婧

利用马尔可夫链进行eeg分类
# Run this code
python classification.py

# 代码结构

data------------------------原始数据，具体解释见data/README.docx

datapreplot.py--------------画图展示原始数据

graph_constructing.py-------根据原始eeg数据，用不同通道之间的相似度作为边权建图，输出label、原始eeg序列数据、图相似度矩阵

markov_chain.py-------------根据label、原始eeg序列数据、图相似度矩阵，提取特征，包括以节点的度为状态的马尔可夫链的转移矩阵、图连通分支数为状态的马尔可夫链的转移矩阵、傅里叶变换状态等。

classification.py-----------根据上游提取的特征进行MLP分类，患病、不患病

