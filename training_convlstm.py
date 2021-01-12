from math import ceil
import os
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
from keras import optimizers
from sklearn.metrics import classification_report
from defs import *
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
import time

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

def split_sequences(sequence, seq_yy, n_steps):
    print("Lunghezza sequenza: ",len(sequence))
    perm = np.random.permutation(len(sequence) - n_steps)  # Shuffle elements

    for i in perm:
        end_ix = i + n_steps
        seq_x, seq_y = sequence[None,i:end_ix], seq_yy[i:end_ix]
        yield np.array(seq_x, dtype=np.float32) , np.array(seq_y, dtype=np.float32)

def build_dataset_train(train_x,train_y):
    # Scrittura del Dataset
    f_t = h5py.File('dati_128/X_train.hdf5', 'w')
    X_train = f_t.create_dataset("train_X", data=train_x)
    f_t.close()

    f = h5py.File('dati_128/y_train.hdf5', 'w')
    y_train = f.create_dataset("y_train", data=train_y)
    f.close()

    return  X_train,y_train

def build_dataset_test(test_x,test_y):
    f = h5py.File('dati_128/X_test.hdf5', 'w')
    X_test = f.create_dataset("X_test", data=test_x)
    f.close()

    f = h5py.File('dati_128/y_test.hdf5', 'w')
    y_test = f.create_dataset("y_test", data=test_y)
    f.close()

    return X_test,y_test


train_dir = "processed/train/"
test_dir = "processed/test/"

'''train_x, train_y = get_data(train_dir)
test_x, test_y= get_data(test_dir)

X_train, y_train = build_dataset_train(train_x,train_y)
X_test, y_test = build_dataset_test(test_x,test_y)'''

#Lettura dei datasets
X_train_f = h5py.File('dati_64/X_train.hdf5', 'r')
X_train = X_train_f.get('train_X').value

Y_train_f = h5py.File('dati_64/y_train.hdf5', 'r')
y_train = Y_train_f.get('y_train').value

X_test_f = h5py.File('dati_64/X_test.hdf5', 'r')
X_test = X_test_f.get('X_test').value

Y_test_f = h5py.File('dati_64/y_test.hdf5', 'r')
y_test = Y_test_f.get('y_test').value

#NON CONSIDERARE - CREAZIONE DI UN SUBSET DI DATI E SALVATAGGIO
'''x_train_sub = X_train[1:20,:,:,:]
y_train_sub = y_train[1:20,]

x_test_sub = X_test[1:5,:,:,:]
y_test_sub = y_test[1:5,]

np.save("x_train_sub.npy",x_train_sub)
np.save("y_train_sub.npy",y_train_sub)
np.save("x_test_sub.npy",x_test_sub)
np.save("y_test_sub.npy",y_test_sub)

'''

'''print("App_train x ",x_train_sub.shape)
print("App_train y",y_train_sub.shape)
print("App_test x ",x_test_sub.shape)
print("App_test y ",y_test_sub.shape)

x_train_sub = np.load("x_train_sub.npy")
y_train_sub = np.load("y_train_sub.npy")
x_test_sub = np.load("x_test_sub.npy")
y_test_sub = np.load("y_test_sub.npy")

X_train = x_train_sub
y_train = y_train_sub
X_test = x_test_sub
y_test = y_test_sub

'''

print("Shape: \n")
print("train_x: ", X_train.shape)
print("y train ",y_train.shape)
print("x_test ",X_test.shape)
print("y test",y_test.shape)

#C'è un problema con le grandi sequenza. Se seleziono una sequenza larga, il SW viene automaticamente killato
seq_len = 15
print("\n\n")

#Utilizzo dimensioni 64x64 come prova. Qunado il training si avvierà, caricherò le 256x256
train_dataset = tf.data.Dataset.from_generator(split_sequences, (tf.float32, tf.float32), output_shapes=([None,15,64, 64, 1], [15,]), args=(X_train, y_train, seq_len))
test_dataset = tf.data.Dataset.from_generator(split_sequences, (tf.float32, tf.float32), output_shapes=([None,15,64, 64, 1], [15,]), args=(X_test, y_test, seq_len))


'''for Xb, yb in train_dataset:
    print(Xb.shape, yb.shape) #Ottengo (15,64,64,1), (15,)
    print(type(train_dataset))
    time.sleep(2)'''

'''X_train, y_train = split_sequences(X_train,y_train, seq_len)
y_train = y_train[:,0]
print("++++***shape dopo lo split [TRAIN X & Y] ",X_train.shape,y_train.shape)


X_test,y_test = split_sequences(X_test,y_test, seq_len)
y_test = y_test[:,0]
print("++++***shape dopo lo split [TEST X & Y] ",X_test.shape,y_test.shape)

X_train = X_train.reshape(X_train.shape[0],seq_len, img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], seq_len, img_width, img_height, 1)
#Provo a stampare la prima immagine alla posizione X(1,5,128,128,1)

print("\n\n")
print("Dopo il reshape [X_train]: ",X_train.shape)
print("Dopo il reshape [X_test]: ",X_test.shape)
print("\n\n")

'''

img_width, img_height = 64,64
input_shape=(seq_len,img_width,img_height,1)
model = MiniModel(input_shape)
model.summary()

sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)

epochs = 60
bs = 16
steps_per_epoch = ceil(10922 / bs)

#opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

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