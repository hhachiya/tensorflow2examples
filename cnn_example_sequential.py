#############################
# tensorflow2のCNNの実装例1（もっとも素人ぽい）
# Sequential APIを用いる場合
#############################
import tensorflow as tf

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
# Sequentialを用いたネットワークの定義
# - addメソッドを用いてlayerインスタンス（Conv2D，BatchNormalization，ReLU，MaxPooling2D，Flatten，Dense，Dropoutなど）をSequentialに追加していく
# - compileメソッドを用いて，最適化方法（adam），損失関数（sparse_categorical_crossentropy），評価方法（accuracy）を設定
def cnn(input_shape):
    model = tf.keras.models.Sequential()

    # conv1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # conv2
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())    
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # conv3
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # fc1
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # fc2
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

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

