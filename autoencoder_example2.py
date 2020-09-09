#############################
# tensorflow2.2のauto-encoderの実装例
# Subclassing APIを用いる場合
#############################
import tensorflow as tf
import matplotlib.pylab as plt
import os
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
# Functionalを用いたネットワークの定義
def autoencoder(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape, name="inputs")

    # conv1
    conv1 = tf.keras.layers.Conv2D(filters=32, strides=(2, 2), padding='same', kernel_size=(3, 3), activation='relu')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

    # conv2
    conv2 = tf.keras.layers.Conv2D(filters=64, strides=(2, 2), padding='same', kernel_size=(3, 3), activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    # fc1
    conv2_flat = tf.keras.layers.Flatten()(conv2)
    fc = tf.keras.layers.Dense(units=64,activation='relu')(conv2_flat)

    # defc
    defc = tf.keras.layers.Dense(units=3136,activation='relu')(fc)
    defc = tf.reshape(defc, tf.shape(conv2))

    # deconv1    
    deconv1 = tf.keras.layers.Conv2DTranspose(filters=64, strides=(2, 2), padding='same', kernel_size=(3, 3), activation='relu')(defc)
    deconv1 = tf.keras.layers.BatchNormalization()(deconv1)

    # deconv2
    deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=(2, 2), padding='same', kernel_size=(3, 3), activation='relu')(deconv1)
    deconv2 = tf.keras.layers.BatchNormalization()(deconv2)    

    # deconv3
    outputs = tf.keras.layers.Conv2DTranspose(filters=1, strides=(1, 1), padding='same', kernel_size=(1, 1), activation='sigmoid')(deconv2)

    return inputs, outputs
#----------------------------

#----------------------------
# Modelクラスを継承し，独自のlayerクラス（myConvとmyFC）を用いてネットワークを定義する
# 独自のモデルクラスを作成
class myModel(tf.keras.Model):

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
        y_pred = self(x, training=False)

        # 損失
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # metricsの更新
        self.compiled_metrics.update_state(y, y_pred)

        # 評価値をディクショナリで返す
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self,data):
        x = data

        # 予測
        return self(x, training=False)     

# モデルの設定
inputs, outputs = autoencoder((H,W,C))
model = myModel(inputs,outputs)

# 学習方法の設定
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
model.summary()

# 中間層の値を取得するためのモデル
features_list = [layer.output for layer in model.layers]
feat_extraction_model = tf.keras.Model(inputs=inputs, outputs=features_list)
#----------------------------

#----------------------------
# 学習
isTrain = True

# 学習したパラメータを保存するためのチェックポイントコールバックを作る
checkpoint_path = "autoencoder_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

if isTrain:
    # fitで学習を実行
    model.fit(x_train, x_train, batch_size=200, epochs=1, callbacks=[cp_callback])
else:
    # 学習したパラメータの読み込み
    model.load_weights(checkpoint_path)
#----------------------------

#----------------------------
# 学習データに対する評価
train_loss, train_mae = model.evaluate(x_train, x_train, verbose=0)
print('Train data loss:', train_loss)
print('Train data mae:', train_mae)
#----------------------------

#----------------------------
# 評価データに対する評価
test_loss, test_mae = model.evaluate(x_test, x_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data mae:', test_mae)
#----------------------------

#----------------------------
# 元画像と復元画像の可視化
img_num = 5
y_test = model.predict_step(x_test[:img_num])
fig = plt.figure()
for i in range(img_num):
    fig.add_subplot(2,img_num,i+1)
    plt.imshow(y_test[i,:,:,0],vmin=0,vmax=1)

for i in range(img_num):
    fig.add_subplot(2,img_num,img_num+i+1)
    plt.imshow(x_test[i,:,:,0],vmin=0,vmax=1)    

plt.tight_layout()
plt.show()
#----------------------------

#----------------------------
# 中間層の値を取得
features = feat_extraction_model(x_test[:img_num])

pdb.set_trace()
#----------------------------
