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
        return x, self.fc2.weights

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

# モデルの設定
model = myModel()
#----------------------------

#----------------------------
# 学習方法の設定
# 学習用と評価用の関数train_stepとtest_stepを定義
# @tf.functionを用いることにより，予測と損失をtensorグラフに繋げることができる
#損失関数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

#最適化関数
optimizer = tf.keras.optimizers.Adam()

# 評価指標
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(x, t):
    with tf.GradientTape() as tape:

        # 予測
        pred, weights = model(x, training=True)

        # 損失
        loss = loss_object(t, pred) + tf.norm(weights[0],axis=0)
    
    # 勾配を用いた学習
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 評価
    train_loss(loss)
    train_accuracy(t, pred)

@tf.function
def test_step(x, t):
    # 予測
    test_pred, test_weights = model(x)

    # 損失
    t_loss = loss_object(t, test_pred) + tf.norm(test_weights[0],axis=0)

    # 評価
    test_loss(t_loss)
    test_accuracy(t, test_pred)
#----------------------------

#----------------------------
# 学習

# ミニバッチの作成
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

for epoch in range(5):
    for images, labels in train_ds:
        train_step(images, labels) #学習

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels) #評価

    print(f"Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}")
    print(f"\ttest-Loss: {test_loss.result()}, test-Accuracy{test_accuracy.result()*100}")
#----------------------------
