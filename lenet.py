###########################
# 라이브러리 사용
import tensorflow as tf
import pandas as pd
 
###########################
# 데이터를 준비합니다. 
(독립, 종속), _ = tf.keras.datasets.cifar10.load_data()
# (독립, 종속),_ = tf.keras.datasets.mnist.load_data()
# 독립 = 독립.reshape(60000, 28, 28, 1)
# mnist는 1차원 이었으니까 종속 변수를 reshape 하지 않는 반면,
# cifar10은 지금 2차원이라서 종속 변수를 reshape 하지 않을 경우 에러가 나기에 해줌 
종속 = pd.get_dummies(종속.reshape(50000))
print(독립.shape, 종속.shape)
 
###########################
# 모델을 완성합니다. 
X = tf.keras.layers.Input(shape=[32, 32, 3])
# padding = 'same' : 추가하게되면 특징맵의 사이즈가 입력 맵과 동일하게 출력된다.
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
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
# 모델을 학습하고
model.fit(독립, 종속, epochs=10)
 
###########################
# 모델을 이용합니다. 
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)
 
# 정답 확인
종속[0:5]
 
# 모델 확인
model.summary()