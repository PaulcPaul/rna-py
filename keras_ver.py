# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation

from carregar_dados import carregar_dados, preparar_iris

raw, data = carregar_dados()
train_x, train_y, test_x, test_y = preparar_iris(data, 60)

num_classes = 3
input_shape = train_x.shape

model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(64, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=15)
score = model.evaluate(test_x, test_y)

print(score)