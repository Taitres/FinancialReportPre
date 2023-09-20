import os
from joblib import dump, load
import numpy as np
from gensim import models, corpora
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

txt_directory = './train_data/'
texts = []
for filename in os.listdir(txt_directory):
    file_path = os.path.join(txt_directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        texts.append(text)



# 创建词典
dictionary = corpora.Dictionary([text.split() for text in texts])

# 创建文档-词汇矩阵
corpus = [dictionary.doc2bow(text.split()) for text in texts]



# 运行LDA模型
num_topics = 10
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# 查看每个主题的词汇分布
topics = lda_model.show_topics(num_topics=num_topics, num_words=10)

# 查看每个文档的主题分布
topic_distributions = [lda_model[doc] for doc in corpus]
print(topic_distributions)
print(len(topic_distributions))
# 创建一个初始的十列数组（全零）
num_topics = 10
num_documents = len(topic_distributions)
array = np.zeros((num_documents, num_topics))

# 将主题分布的值填充到数组中
for i, doc_topics in enumerate(topic_distributions):
    for topic_id, weight in doc_topics:
        array[i, topic_id] = weight



labels = pd.read_csv('./train_labels.csv')['label'].tolist()

X_train = array  # 特征，主题分布
y_train = labels[:495]  # 目标变量，分类标签
y_train = np.array(y_train)

n_splits = 2  # 使用5折交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []
i = 1
for train_idx, val_idx in kf.split(X_train):

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]


    model = DecisionTreeClassifier(max_depth=8)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_pred_train = model.predict(X_tr)
    r2_pre = f1_score(y_val, y_pred)
    r2_train = f1_score(y_tr, y_pred_train)
    print(f"第{i}折,验证集:", r2_pre, "训练集:", r2_train)
    i += 1
    scores.append(r2_pre)


model = DecisionTreeClassifier(max_depth=8)
model.fit(X_train, y_train)

# 保存模型到文件
dump(model, 'model/my-model.joblib')