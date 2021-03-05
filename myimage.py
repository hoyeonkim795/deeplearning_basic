###########################
# 라이브러리 로딩
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
 
###########################
# 이미지 읽어서 데이터 준비하기
paths = glob.glob('./notMNIST_small/*/*.png')
paths = np.random.permutation(paths)
독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
종속 = np.array([paths[i].split('/')[-2] for i in range(len(paths))])
print(독립.shape, 종속.shape)
 
독립 = 독립.reshape(18724, 28, 28, 1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)
 
###########################
# 모델을 완성합니다. 
X = tf.keras.layers.Input(shape=[28, 28, 1])
 
H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)
 
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)
 
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
 
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
 
###########################
# 모델을 학습
model.fit(독립, 종속, epochs=10)
 
###########################
# 모델을 이용합니다. 
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)
 
# 정답 확인
종속[0:5]
 
# 모델 확인
model.summary()