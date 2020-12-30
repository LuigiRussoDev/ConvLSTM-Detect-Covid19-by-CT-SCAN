from math import ceil
import os
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
from keras import optimizers
from sklearn.metrics import classification_report
from defs import *
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def split_sequences(sequence, seq_yy, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], seq_yy[i:end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X) , np.array(y)

train_dir = "processed/train/"
test_dir = "processed/test/"


'''train_x, train_y = get_data(train_dir)
test_x, test_y= get_data(test_dir)'''

#Scrittura del Dataset
'''f = h5py.File('dati_64/X_train.hdf5', 'w')
X_train = f.create_dataset("train_X",data=train_x)

f = h5py.File('dati_64/y_train.hdf5', 'w')
y_train = f.create_dataset("y_train",data=train_y)

f = h5py.File('dati_64/X_test.hdf5', 'w')
X_test = f.create_dataset("X_test",data=test_x)

f = h5py.File('dati_64/y_test.hdf5', 'w')
y_test = f.create_dataset("y_test",data=test_y)'''

#Lettura dei datasets
X_train_f = h5py.File('dati_64/X_train.hdf5', 'r')
X_train = X_train_f['train_X']

Y_train_f = h5py.File('dati_64/y_train.hdf5', 'r')
y_train = Y_train_f['y_train']

X_test_f = h5py.File('dati_64/X_test.hdf5', 'r')
X_test = X_test_f['X_test']

Y_test_f = h5py.File('dati_64/y_test.hdf5', 'r')
y_test = Y_test_f['y_test']




print("Shape: \n")
print("train_x: ", X_train.shape)
print("y train ",y_train.shape)
print("x_test ",X_test.shape)
print("y test",y_test.shape)

seq_len = 10

print("\n\n")
X_train, y_train = split_sequences(X_train,y_train, seq_len)
y_train = y_train[:,0]
#print("Y train ",y_train[1:100])
print("++++***shape dopo lo split [TRAIN X & Y] ",X_train.shape,y_train.shape)
#print(y_train[1:100])

X_test,y_test = split_sequences(X_test,y_train, seq_len)
y_test = y_test[:,0]
#print("Y test ",y_test[1:100])
print("++++***shape dopo lo split [TEST X & Y] ",X_test.shape,y_test.shape)


img_width, img_height = 64, 64

X_train = X_train.reshape((X_train.shape[0], seq_len, img_width, img_height, 1))
X_test = X_test.reshape((X_test.shape[0], seq_len, img_width, img_height, 1))

print("\n\n")
print("Dopo il reshape [X_train]: ",X_train.shape)
print("Dopo il reshape [X_test]: ",X_test.shape)
print("\n\n")


input_shape=(seq_len,img_width,img_height,1)
model = MiniModel(input_shape)
model.summary()

sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)

epochs = 60
bs = 16
steps_per_epoch = ceil(10922 / bs)

#opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
path_checkpoint = "best_model.hdf5"

#TODO: CHECKPOINT SOLO COME TEST. DA COMPLETARE IN CASO DI RESTORE DEL TRAINING
checkpointer = ModelCheckpoint(path_checkpoint, verbose=1,
    save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
model.save(path_checkpoint.format(epoch=0))

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