import keras
from math import ceil
import os
from sklearn.metrics import confusion_matrix
from keras import optimizers
from sklearn.metrics import classification_report
from defs import *


os.environ["CUDA_VISIBLE_DEVICES"]="1"


def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


train_dir = "processed/train/"
test_dir = "processed/test/"

img_height , img_width = 32, 32
seq_len = 5

'''train_x, train_y = get_data(train_dir)
test_x, test_y= get_data(test_dir)

np.save('X_train.npy',train_x)
np.save('y_train.npy',train_y)
np.save('y_test.npy',test_y)
np.save('X_test.npy',test_x)'''

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
X_test = np.load('X_test.npy')


print("New Shape: \n")
print("train_x: ", X_train.shape)
print("y train ",y_train.shape)
print("x_test ",X_test.shape)
print("y test",y_test.shape)

X_train,y_train = split_sequences(X_train, seq_len)
print("++++***shape dopo lo split [TRAIN] ",X_train.shape,y_train.shape)


X_test,y_test = split_sequences(X_test, seq_len)
print("++++***shape dopo lo split [TEST] ",X_test.shape,y_test.shape)
y_train = y_train[:,0,0,0]
y_test = y_test[:,0,0,0]



# normalize the data
'''X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train - X_train.mean()
X_test = X_test - X_test.mean()
X_train = X_train / X_train.std(axis=0)
X_test = X_test / X_test.std(axis=0)'''


y_train = y_train - 1
y_test = y_test - 1
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]


n_steps, n_length = 5, 32
X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
'''
ottengo:
X_train = 21821,5,1,32,32
y_train = 21821,2
test x = 9347,5,1,32,32
'''


input_shape=(n_steps,1,n_length,n_features)
model = MiniModel(input_shape)

model.summary()
print("->>> trainX: ",X_train.shape,"testX ",X_test.shape,"N features ",n_features)
print("->>> testY: ",y_test.shape," train Y",y_train.shape)
print("n timesteps: ",n_timesteps,"n features ",n_features,"n output ",n_outputs)


sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

epochs = 60
bs = 16
steps_per_epoch = ceil(10922 / bs)

#opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs,batch_size=bs,validation_data=(X_test,y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('accuracy_convlstm.png')

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_convlstm.png')


accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)

target_names = ['Normal', 'Covid']

y_pred = model.predict(X_test)

'''fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)'''

Y_true = np.argmax(y_test, axis=1)
Y_pred_classes = np.argmax(y_pred, axis=1)

roc_each_classes(y_test,y_pred)


print('\n', classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1),
                                  target_names=target_names))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes=target_names)

print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])

model.save("covid19.h5")