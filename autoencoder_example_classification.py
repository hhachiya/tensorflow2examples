#############################
# tensorflow2.2のauto-encoder（binary-classification）の実装例
# Subclassing APIを用いる場合
#############################
import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy

#----------------------------
# 学習モード
isTrain = True

# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C, O = 28, 28, 1, 2

# MNISTデータの読み込み
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # 画像の正規化
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# 画像の2値化
x_train[x_train<125] = 0
x_train[x_train>=125] = 1
x_test[x_test<125] = 0
x_test[x_test>=125] = 1

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# （データ数，高さ，幅，チャネル数）にrehspae
x_train = x_train.reshape(x_train.shape[0], H, W, C)
x_test = x_test.reshape(x_test.shape[0], H, W, C)

# ラベルをone-hot表現にする
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]
#----------------------------

#----------------------------
# Subclassingを用いたネットワークの定義

# Layerクラスを継承して独自のconvolution用のレイヤークラスを作成
class myConv(tf.keras.layers.Layer):
    def __init__(self, chn=32, conv_kernel=(3,3), strides=(2,2), pool_kernel=(2,2), activation='relu', isBatchNorm=True, isPool=False):
        super(myConv, self).__init__()
        self.activation = activation
        self.isBatchNorm = isBatchNorm
        self.isPool = isPool

        self.conv_relu = tf.keras.layers.Conv2D(filters=chn, strides=strides, padding='same', kernel_size=conv_kernel, activation='relu')
        self.conv_sigmoid =  tf.keras.layers.Conv2D(filters=chn, strides=strides, padding='same', kernel_size=conv_kernel, activation='sigmoid')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_kernel) 

    def call(self, x):
        if self.activation == 'relu':
            x = self.conv_relu(x)
        elif self.activation == 'sigmoid':
            x = self.conv_sigmoid(x)

        if self.isBatchNorm:
            x = self.batchnorm(x)

        if self.isPool:
            x = self.pool(x)

        return x

# Layerクラスを継承して独自のdeconvolution用のレイヤークラスを作成
class myDeconv(tf.keras.layers.Layer):
    def __init__(self, chn=32, conv_kernel=(3,3), strides=(2,2), activation='relu', isBatchNorm=True):
        super(myDeconv, self).__init__()
        self.activation = activation
        self.isBatchNorm = isBatchNorm

        self.conv_relu = tf.keras.layers.Conv2DTranspose(filters=chn, strides=strides, padding='same', kernel_size=conv_kernel, activation='relu')
        self.conv_sigmoid =  tf.keras.layers.Conv2DTranspose(filters=chn, strides=strides, padding='same', kernel_size=conv_kernel, activation='sigmoid')
        self.conv =  tf.keras.layers.Conv2DTranspose(filters=chn, strides=strides, padding='same', kernel_size=conv_kernel)
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        if self.activation == 'relu':
            x = self.conv_relu(x)
        elif self.activation == 'sigmoid':
            x = self.conv_sigmoid(x)
        elif self.activation == 'softmax':
            x = self.conv(x)
            x = tf.keras.activations.softmax(x, axis=3)
        elif self.activation == 'none':
            x = self.conv(x)

        if self.isBatchNorm:
            x = self.batchnorm(x)        

        return x

# Layerクラスを継承して独自のFC用のレイヤークラスを作成
class myFC(tf.keras.layers.Layer):
    def __init__(self, chn=10, activation='relu', isFlat=False):
        super(myFC, self).__init__()
        self.activation = activation
        self.isFlat = isFlat

        self.flatten = tf.keras.layers.Flatten()
        self.fc_relu = tf.keras.layers.Dense(units=chn, activation='relu')
        self.fc_sigmoid = tf.keras.layers.Dense(units=chn, activation='sigmoid')

    def call(self, x):
        if self.isFlat:
            x = self.flatten(x)

        if self.activation == 'relu':
            x = self.fc_relu(x)
        elif self.activation == 'sigmoid':
            x = self.fc_sigmoid(x)
        return x

# Modelクラスを継承し，独自のネットワークを定義する
class myModel(tf.keras.Model):
    def __init__(self,isEmbedLabel=True):
        super(myModel, self).__init__()
        self.isEmbedLabel = isEmbedLabel

        # conv, fc, defc & deconv layers
        self.conv1 = myConv(chn=32, conv_kernel=(3,3), activation='relu')
        self.conv2 = myConv(chn=64, conv_kernel=(3,3), activation='relu')
        self.fc = myFC(chn=64, activation='relu', isFlat=True)
        self.defc = myFC(chn=3136, activation='relu')
        self.deconv1 = myDeconv(chn=64, conv_kernel=(3,3))
        self.deconv2 = myDeconv(chn=32, conv_kernel=(3,3))
        self.deconv3 = myDeconv(chn=2, conv_kernel=(3,3), strides=(1,1), activation='softmax', isBatchNorm=False)

    def call(self, data):
        (x,y) = data

        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        z = self.fc(conv2)
        
        if self.isEmbedLabel:
            # embeddingにone-hotラベルをconcat
            zz = tf.concat([z,y], axis=1)
        else:
            zz = z     

        # decoder
        defc = self.defc(zz)
        defc_reshape = tf.reshape(defc,tf.shape(conv2))
        deconv1 = self.deconv1(defc_reshape)
        deconv2 = self.deconv2(deconv1)
        output = self.deconv3(deconv2)

        return output, (conv1,conv2,z,zz,defc,defc_reshape,deconv1,deconv2)

    # 学習
    def train_step(self,data):
        x, y_true = data
        
        # 入力を画像とラベルに分解
        x, label = x

        with tf.GradientTape() as tape:
            
            # 予測
            y_pred, _ = self((x,label), training=True)
            
            # 損失
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        
        # 勾配を用いた学習
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 評価値の更新
        self.compiled_metrics.update_state(y_true, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

    # 評価
    def test_step(self, data):
        x, y_true = data
        
        # 入力を画像とラベルに分解
        x, label = x

        # 予測
        y_pred, _ = self((x,label), training=False)

        # 損失
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # metricsの更新
        self.compiled_metrics.update_state(y_true, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

    # 予測
    def predict_step(self,data):
        x = data

        # 入力を画像とラベルに分解
        x, label = x

        # 予測
        return self((x,label), training=False)     

# モデルの設定
model = myModel(isEmbedLabel=True)

# 学習方法の設定
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#----------------------------

#----------------------------
# 学習
# 学習したパラメータを保存するためのチェックポイントコールバックを作る
checkpoint_path = "autoencoder_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

if isTrain:
    # fitで学習を実行
    history = model.fit((x_train, y_train_onehot), x_train, batch_size=200, epochs=5, validation_split=0.1, callbacks=[cp_callback])

    # 損失のプロット
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.arange(len(acc))

    plt.plot(epochs,acc,'bo-',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.title('Training and validation acc')
    plt.legend()

    plt.figure()
    plt.plot(epochs,loss,'bo-',label='training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()    
else:
    # 学習したパラメータの読み込み
    model.load_weights(checkpoint_path)
#----------------------------

#----------------------------
# 学習データに対する評価
train_loss, train_acc = model.evaluate((x_train[:1000], y_train_onehot[:1000]),x_train[:1000], verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_acc)
#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_acc = model.evaluate((x_test, y_test_onehot), x_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_acc)
#----------------------------

#----------------------------
# 元画像と復元画像の可視化
img_num = 5
y_pred_test, z_test = model.predict_step((x_test[:img_num], y_test_onehot[:img_num]))
fig = plt.figure()
for i in range(img_num):
    fig.add_subplot(2,img_num,i+1)    
    plt.imshow(y_pred_test[i,:,:,1],vmin=0,vmax=1)

for i in range(img_num):
    fig.add_subplot(2,img_num,img_num+i+1)
    plt.imshow(x_test[i,:,:,0],vmin=0,vmax=1)    

plt.tight_layout()
plt.show()
#----------------------------

pdb.set_trace()
