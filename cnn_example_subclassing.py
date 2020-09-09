#############################
# tensorflow2のCNNの実装例3（もっとも玄人ぽい）
# Subclassing APIを用いる場合
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
# Subclassingを用いたネットワークの定義

# Layerクラスを継承して独自のconvolution用のレイヤークラスを作成
class myConv(tf.keras.layers.Layer):
    def __init__(self,chn=32, conv_kernel=(3,3), pool_kernel=(2,2), isPool=True):
        super(myConv, self).__init__()
        self.isPool = isPool

        self.conv = tf.keras.layers.Conv2D(chn, conv_kernel)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_kernel)        

    def call(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        if self.isPool:
            x = self.pool(x)
        return x

# Layerクラスを継承して独自のFC用のレイヤークラスを作成
class myFC(tf.keras.layers.Layer):
    def __init__(self, hidden_chn=64, out_chn=10):
        super(myFC, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(hidden_chn, activation='relu')
        self.fc2 = tf.keras.layers.Dense(out_chn, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Modelクラスを継承し，独自のlayerクラス（myConvとmyFC）を用いてネットワークを定義する
# 独自のモデルクラスを作成
class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = myConv(chn=32, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv2 = myConv(chn=64, conv_kernel=(3,3), pool_kernel=(2,2))
        self.conv3 = myConv(chn=64, conv_kernel=(3,3), isPool=False)
        self.fc = myFC(hidden_chn=64, out_chn=10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

    def train_step(self,data):
        x, y = data

        with tf.GradientTape() as tape:
            
            # 予測
            y_pred = self(x, training=True)

            # 損失
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # 勾配を用いた学習
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 評価値の更新
        self.compiled_metrics.update_state(y, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        # 予測
        pred = self(x, training=True)

        # 損失
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # metricsの更新
        self.compiled_metrics.update_state(y, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

# モデルの設定
model = myModel()

# 学習方法の設定
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#----------------------------

#----------------------------
# 学習
# - cnn関数を実行しネットワークを定義
# - fitで学習を実行
model.fit(x_train, y_train, batch_size=200, epochs=1)
#----------------------------

#----------------------------
# 学習データに対する評価
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_accuracy)
#----------------------------

#----------------------------
# 学習データに対する評価
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_accuracy)
#----------------------------