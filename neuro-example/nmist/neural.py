import pandas as pd
import keras as kr

from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda

import numpy as np


train = pd.read_csv("./data/train.csv")
print("TRAIN.CSV")
print(train.shape)
print(train.head())
test = pd.read_csv("./data/test.csv")
print("TEST.CSV")
print(test.shape)
print(test.head())

X_train = train.ix[:,1:].values.astype('float32')  # all pixel values
y_train = train.ix[:,0].values.astype('int32')  # only labels i.e targets digits
X_test = test.values.astype('float32')

mean_px = np.concatenate((X_train, X_test)).mean().astype(np.float32)
std_px = np.concatenate((X_train, X_test)).std().astype(np.float32)

X_train = (X_train - mean_px) / std_px
X_test = (X_test - mean_px) / std_px


print("X_TRAIN")
print(X_train[0:5])
print("Y_TRAIN")
print(y_train[0:5])
print("X_TEST")
print(X_test[0:5])

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = kr.utils.to_categorical(y_train, num_classes=10)
print("ONE_HOT_LABELS")
print(one_hot_labels)


model.fit(X_train, one_hot_labels, epochs=20, batch_size=10, verbose=2)

model.save("./modelstate")

model = kr.models.load_model("./modelstate")

score = model.predict(X_test, verbose=2)
score_max = np.argmax(score, axis=1)
print("SCORE")
print(score)

submissions = pd.DataFrame({"ImageId": list(range(1,len(score_max)+1)),
                         "Label": score_max})
submissions.to_csv("DR.csv", index=False, header=True)

pass







