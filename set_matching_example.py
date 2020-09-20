#############################
# tensorflow2.2のauto-encoder（binary-classification）の実装例
# Subclassing APIを用いる場合
#############################
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy
import pickle

#----------------------------
# パラメータの作成
isTrain = True

isSelfAttention = True

# ベースチャネル数
baseChn = 32

# データの作成
isMakeData = 0

# データの種類
dataMode = 2 #順不同

if dataMode == 1:
    dataFileName = 'MNIST_group_permutate.pkl'
elif dataMode == 2:
    dataFileName = 'MNIST_group.pkl'

gID = np.array([[1,3],[2,4]])
#----------------------------

#----------------------------
# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C = 28, 28, 1

if isMakeData:
    # MNISTデータの読み込み
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # アイテム数が２つのグループを作成
    def makeGroup(X1,X2,dataMode=1):
        if dataMode == 1:
            #半分はX1とX2の順番を変える
            X = np.array([np.concatenate([X1[[i]],X2[[i]]],axis=0) if i%2==0 else np.concatenate([X2[[i]],X1[[i]]],axis=0) for i in range(np.min([len(X1),len(X2)]))]) 
        elif dataMode == 2:
            # 順番固定
            X = np.array([np.concatenate([X1[[i]],X2[[i]]],axis=0) for i in range(np.min([len(X1),len(X2)]))]) 
        return X

    X1_train = makeGroup(x_train[y_train==gID[0,0]],x_train[y_train==gID[0,1]],dataMode=2)
    X1_test = makeGroup(x_test[y_test==gID[0,0]],x_test[y_test==gID[0,1]],dataMode=2)
    X2_train = makeGroup(x_train[y_train==gID[1,0]],x_train[y_train==gID[1,1]],dataMode=2)
    X2_test = makeGroup(x_test[y_test==gID[1,0]],x_test[y_test==gID[1,1]],dataMode=2)

    x_train = np.concatenate([X1_train,X2_train],axis=0)
    y_train = np.concatenate([np.zeros(len(X1_train)),np.ones(len(X2_train))])
    x_test = np.concatenate([X1_test,X2_test],axis=0)
    y_test = np.concatenate([np.zeros(len(X1_test)),np.ones(len(X2_test))])

    inds_train = np.random.permutation(len(x_train))
    inds_test = np.random.permutation(len(x_test))
    x_train = x_train[inds_train]
    y_train = y_train[inds_train]
    x_test = x_test[inds_test]
    y_test = y_test[inds_test]


    with open(dataFileName,'wb') as fp:
        pickle.dump(x_train,fp)
        pickle.dump(y_train,fp)
        pickle.dump(x_test,fp)
        pickle.dump(y_test,fp)

else:
    # MNISTデータの読み込み
    with open(dataFileName,'rb') as fp:
        x_train = pickle.load(fp)
        y_train = pickle.load(fp)
        x_test = pickle.load(fp)
        y_test = pickle.load(fp)

# 集合の中のアイテム数（固定でいいのか？）
nItem = x_train.shape[1]

# 画像の正規化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# （データ数，高さ，幅，チャネル数）にrehspae
x_train = x_train.reshape(x_train.shape[0], nItem, H, W, C)
x_test = x_test.reshape(x_test.shape[0], nItem, H, W, C)
#----------------------------

#----------------------------
# CS function to compute cross set matching scores
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self,dim=20, nHead=1):
        super(cross_set_score, self).__init__()
        self.dim = dim
        self.nHead = nHead

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_nHead) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_nHead outputs dim*nHead-dim vector
        self.linear = tf.keras.layers.Dense(units=self.dim*self.nHead,activation='relu',use_bias=False)

        # linear function to combine multi-head score maps
        self.conv = tf.keras.layers.Conv2D(filters=1, strides=(1,1), padding='same', kernel_size=(1,1))

    def call(self, x):

        # linear transofrmation (B*nItem,dim*nHead)
        x = self.linear(x)

        # reshape (B*nItem,dim*nHead) to (nHead, B*nItem,dim)
        x = tf.transpose(tf.reshape(x,[-1,self.nHead,self.dim]),[1,0,2])

        # inner products between all pairs of items, outputing (nhead, nSet*nItem, nSet*nItem)-score map
        inner_prods = tf.matmul(x,tf.transpose(x,[0,2,1]))/tf.sqrt(tf.cast(self.dim,tf.float32))

        # reshape (nHead, nSet*nItem, nSet*nItem,1)
        inner_prods = tf.expand_dims(inner_prods,-1)

        # sum up score-map in each block of size (nItem, nItem) using conv2d with weights of ones
        # outputing (nHead, nSet, nSet, 1)-score map
        scores = tf.keras.layers.Conv2D(filters=1,strides=(nItem,nItem),kernel_size=(nItem,nItem),
                    trainable=False,use_bias=False,weights=[tf.ones((nItem,nItem,1,1))])(inner_prods)

        # devided by the two numbers of items of two sets（NOTE that nItem is SIMPLY fixed）
        scores = scores/nItem/nItem

        # linearly combine multi-head score maps
        # reshape (nHead, nSet, nSet, 1) to (1, nSet, nSet, nHead)
        scores = tf.transpose(scores,[3,1,2,0])
        scores = self.conv(scores)

        return scores[0,:,:,0]
#----------------------------        

#----------------------------
# 全体のネットワーク
class myModel(tf.keras.Model):
    def __init__(self, self_num_heads=2):
        super(myModel, self).__init__()

        # conv, fc, defc & deconv layers
        self.conv1 = tf.keras.layers.Conv2D(filters=baseChn, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.cross_set_score = cross_set_score()
        self.self_attention = tfa.layers.MultiHeadAttention(head_size=baseChn*2,num_heads=self_num_heads)
        self.fc = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, x):

        # reshape (B, nItem, H, W, C) to (B*nItem, H, W, C)
        x = tf.reshape(x,[-1,H,W,C])
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.globalpool(x)
        
        # encoder (self-attention)
        if isSelfAttention:
            # reshape (B*nItem, D) to (B, nItem, D)
            x = tf.reshape(x,[-1,nItem,tf.shape(x)[1]])
            x = self.self_attention([x,x]) # [Queries, Values]

            # reshape (B, nItem, D) to (B*nItem, D)
            x = tf.reshape(x,[-1,tf.shape(x)[2]])

        # decoder (cross-set transofrmation:cseft)            

        # cross set matching score
        score = self.cross_set_score(x)

        # classificaiton
        output = self.fc(tf.reshape(score,[-1,1]))
   
        return output, score

    # 星取表（ラベルyが一致する場合1、異なる場合0）を作成
    def cross_set_label(self, y):
        # 星取表の行
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])

        # 星取表の列        
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # ラベルが一致する場合1、異なる場合0
        return 1-tf.abs(y_rows - y_cols)

    # def toBinaryLabel(self,y):
    #     dNum = tf.shape(y)[0]
    #     y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

    #     return tf.reshape(y,[dNum,-1])

    # 学習
    def train_step(self,data):
        x, y_true = data
        
        with tf.GradientTape() as tape:
            
            # 予測
            y_pred, _ = self(x, training=True)

            # クロスマッチラベルに変換
            y_true = self.cross_set_label(y_true)
            y_true = tf.reshape(y_true,-1)

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
        
        # 予測
        y_pred, _ = self(x, training=False)

        # クロスマッチラベルに変換
        y_true = self.cross_set_label(y_true)
        y_true = tf.reshape(y_true,-1)

        # 損失
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # metricsの更新
        self.compiled_metrics.update_state(y_true, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

    # 予測
    def predict_step(self,data):
        x = data

        # 予測
        y_pred, _ = self(x, training=False)

        # 予測
        return y_pred

# モデルの設定
model = myModel()

# 学習方法の設定
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],run_eagerly=True)
#----------------------------

#----------------------------
# 学習
# 学習したパラメータを保存するためのチェックポイントコールバックを作る
checkpoint_path = "setmatching_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


if isTrain:

    # fitで学習を実行
    history = model.fit(x_train, y_train, batch_size=10, epochs=10, validation_split=0.2, callbacks=[cp_callback])

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
train_loss, train_acc = model.evaluate(x_train[:1000], y_train[:1000], verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_acc)
#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_acc)
#----------------------------

y_pred,score=model(x_train[:4])

pdb.set_trace()