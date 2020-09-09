#############################
# tensorflow2のCNNの実装例2（tensorflow 1系ぽい）
# Functional APIを用いる場合
#############################
import tensorflow as tf
import pdb

#----------------------------
# データの作成
# 画像サイズ（高さ，幅，チャネル数）
H, W, C = 28, 28, 1

# MNISTデータの読み込み
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 画像の正規化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# （データ数，高さ，幅，チャネル数）にrehspae
x_train = x_train.reshape(x_train.shape[0], H, W, C)
x_test = x_test.reshape(x_test.shape[0], H, W, C)
#----------------------------

#----------------------------
# Tensor変数（tensorflow 1系風）を用いたネットワークの定義
# - Input()を用いてTensorの初期化
# - Conv2D()，BatchNormalization()，ReLU()，MaxPooling2D()，Flatten()，Dense()，Dropout()などを用いてTensorのグラフ構造を定義
# - Model()を用いて，入力inputsから出力outputsまでのTensorのグラフ構造（ネットワーク）をインスタンス化
# - compileメソッドを用いて，最適化方法（adam），損失関数（sparse_categorical_crossentropy），評価方法（accuracy）を設定
def cnn(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape, name="inputs")

    # conv1
    conv1 = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    conv1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)

    # conv2
    conv2 = tf.keras.layers.Conv2D(64, (3, 3))(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    conv2 = tf.keras.layers.MaxPool2D((2, 2))(conv2)

    # conv3
    conv3 = tf.keras.layers.Conv2D(64, (3, 3))(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    
    # fc1
    conv3_flat = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(64,activation='relu')(conv3_flat)
    
    # fc2
    outputs = tf.keras.layers.Dense(10,activation='softmax')(fc1)

    # モデルの設定
    model = tf.keras.Model(inputs, outputs)

    # 学習方法の設定
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model
#----------------------------

#----------------------------
# 学習
# - cnn関数を実行しネットワークを定義
# - fitで学習を実行
model = cnn((H,W,C))
model.summary()
model.fit(x_train, y_train, batch_size=200, epochs=2)
#----------------------------

#----------------------------
# 学習データに対する評価
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_accuracy)
#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_accuracy)
#----------------------------


