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
import sys

#============================================================
# tensorflow2.xでのGPUの設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # 
    for k in range(len(physical_devices)):
        tf.config.set_visible_devices(physical_devices[k], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
#============================================================

#----------------------------
# set parameters
isTrain = True

# flag of self-attention and cseft
isPretrain = int(sys.argv[1])
isSelfAttention = int(sys.argv[2])
isCseft = int(sys.argv[3])

# base channel numbers of CNN layer
baseChn = 32

# flag of making data
isMakeData = 0

# batch size
batch_size = 10

epochs = 3

# set data mode
dataMode = 2 # 1: reverse the order, 2: fix the order

if dataMode == 1:
    dataFileName = 'MNIST_group_permutate.pkl'
elif dataMode == 2:
    dataFileName = 'MNIST_group.pkl'

gID = np.array([[1,3],[2,4]])
#----------------------------

#----------------------------
# make data
# image size (height, weidth, channels)
H, W, C = 28, 28, 1

if isMakeData:
    # load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # make two different sets containing two images (items) of 2&4 or 1&3
    def makeGroup(X1,X2,dataMode=1):
        if dataMode == 1:
            # reverse the order of images (items)
            X = np.array([np.concatenate([X1[[i]],X2[[i]]],axis=0) if i%2==0 else np.concatenate([X2[[i]],X1[[i]]],axis=0) for i in range(np.min([len(X1),len(X2)]))]) 
        elif dataMode == 2:
            # fix the order of images (items)
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
    # load data from pickle file
    with open(dataFileName,'rb') as fp:
        x_train = pickle.load(fp)
        y_train = pickle.load(fp)
        x_test = pickle.load(fp)
        y_test = pickle.load(fp)


# the number of items in a set (NOTE that the number is FIXED)
nItem = x_train.shape[1]

# normalize images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape to（N，H，W，C）
x_train = x_train.reshape(x_train.shape[0], nItem, H, W, C)
x_test = x_test.reshape(x_test.shape[0], nItem, H, W, C)
#----------------------------

#----------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        #self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False)
        self.linear2 = tf.keras.layers.Dense(1,use_bias=False)

    def call(self, x):
        nSet = tf.shape(x)[0]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        # linear transofrmation from (nSet, nSet, nItem, Xdim) to (nSet, nSet, nItem, head_size*num_heads)
        x = self.linear(x)

        # reshape (nSet*nItem, nSet, head_size*num_heads) to (num_heads, nSet*nItem, nSet, head_size)
        x = tf.transpose(tf.reshape(x,[nSet, nSet, nItem, self.num_heads, self.head_size]),[0,1,3,2,4])        

        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[0,1] and x[1,0] are extracted to compute inner product when nItem=2
        # More generally, between set #i and set #j, cseft x[i,j] and x[j,i] are extracted.
        # Outputing (nSet, nSet, num_heads)-score map
        scores = tf.stack(
            [[
                tf.reduce_sum(tf.reduce_sum(
                tf.keras.layers.ReLU()(tf.matmul(x[j,i],tf.transpose(x[i,j],[0,2,1]))/sqrt_head_size)
                ,axis=1),axis=1)
                for i in range(nSet)] for j in range(nSet)]
             )

        # devided by the two numbers of items of two sets（NOTE that nItem is SIMPLY fixed）
        scores = scores/nItem/nItem

        # linearly combine multi-head score maps (nSet, nSet, num_heads) to (nSet, nSet)
        scores = self.linear2(scores)[:,:,0]

        return scores
#----------------------------         

#----------------------------
# cross-set feature (cseft)
class set_transofrm(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2, activation="relu", self_attention=False):
        super(set_transofrm, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads        
        self.activation = activation
        self.self_attention = self_attention

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        self.linear1 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_transform')
        self.linear2 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_transform')
        self.linear3 = tf.keras.layers.Dense(units=self.head_size*self.num_heads, use_bias=False, name='set_transform')
        self.linear4 = tf.keras.layers.Dense(units=self.head_size, use_bias=False, name='set_transform')

    def call(self, x, y):
        # number of sets
        nSet = tf.shape(x)[0]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        # reshape to (nSet, nItem, dim)
        # linear transofrmation (nSet*nItem, nItem, head_size*num_heads)
        x = self.linear1(x)
        y1 = self.linear2(y)
        y2 = self.linear3(y)

        # # reshape (nSet, nItem, head_size*num_heads) to (nSet, num_heads, nItem, head_size)
        x = tf.transpose(tf.reshape(x,[nSet, nItem, self.num_heads, self.head_size]),[0,2,1,3])
        y1 = tf.transpose(tf.reshape(y1,[nSet, nItem, self.num_heads, self.head_size]),[0,2,1,3])
        y2 = tf.transpose(tf.reshape(y2,[nSet, nItem, self.num_heads, self.head_size]),[0,2,1,3])

        # inner products between all pairs of items, outputing (nSet, num_heads, nSet*nItem, nSet*nItem)-score map        
        xy1 = tf.matmul(x,tf.transpose(y1,[0,1,3,2]))/sqrt_head_size

        if self.activation=='softmax':
            # normalized by softmax
            attention_weight = tf.nn.softmax(xy1,axis=-1)
        elif self.activation=='relu':
            # non-negative using Relu
            attention_weight = tf.keras.layers.ReLU()(xy1)
            attention_weight = attention_weight/nItem
        
        if not self.self_attention:
            # reshape (nSet, num_heads, nItem, head_size) to (nSet*nSet, num_heads, nItem, head_size)
            attention_weight = tf.tile(attention_weight,[nSet,1,1,1])
            y2 = tf.repeat(y2,nSet,axis=0)
                    
        # computing weighted y2, outputing (nSet, num_heads, nItem, head_size)
        weighted_y2s = tf.matmul(attention_weight, y2)

        # reshape (nSet, num_heads, nItem, head_size) to (nSet, nItem, head_size*num_heads)
        weighted_y2s = tf.reshape(tf.transpose(weighted_y2s,[0,2,1,3]),[-1, nItem, self.num_heads*self.head_size])

        # combine multi-head to (nSet, nItem, head_size)
        output = self.linear4(weighted_y2s)

        if not self.self_attention:
            # reshape to (nSet, nSet, nItem, head_size)
            output = tf.reshape(output, [nSet, nSet, nItem, self.head_size])

        return output
#----------------------------  

#----------------------------
# design network architecture
class myModel(tf.keras.Model):
    def __init__(self, isPretrain=True, isSelfAttention=True, isCseft=True, num_heads=2):
        super(myModel, self).__init__()
        self.isSelfAttention = isSelfAttention
        self.isCseft = isCseft
        self.isPretrain = isPretrain

        # conv, fc, defc & deconv layers
        self.conv1 = tf.keras.layers.Conv2D(filters=baseChn, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.cross_set_score = cross_set_score(head_size=baseChn*2, num_heads=num_heads)
        self.self_attention = set_transofrm(head_size=baseChn, num_heads=num_heads, activation="softmax", self_attention=True)
        self.cseft = set_transofrm(head_size=baseChn, num_heads=num_heads, activation="softmax")
        self.fc1 = tf.keras.layers.Dense(2, activation='softmax', name='class')
        self.fc2 = tf.keras.layers.Dense(baseChn, activation='relu', name='setmatching')
        self.fc3 = tf.keras.layers.Dense(2, activation='softmax', name='setmatching')

    def call(self, x):

        debug = []

        # reshape (nSet, nItem, H, W, C) to (nSet*nItem, H, W, C)
        nSet = tf.shape(x)[0]
        x = tf.reshape(x,[-1,H,W,C])
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.globalpool(x)
        debug.append(x)

        # classificaiton of set
        output1 = self.fc1(tf.reshape(x,[nSet,-1]))

        # reshape (nSet*nItem, D) to (nSet, nItem, D)
        x = tf.reshape(x,[nSet, nItem, -1])

        # Attention
        if self.isSelfAttention:
            # encoder (self-attention)
            # input: (nSet, nItem, D), output:(nSet, nItem, D)
            x = self.self_attention(x,x)
            debug.append(x)

        if self.isCseft:
            # decoder (cross-set transofrmation:cseft)
            # input: (nSet, nItem, D)
            # output:(nSet, nSet, nItem, D)
            x = self.cseft(x,x)
            debug.append(x)
        else:
            x = tf.tile(tf.expand_dims(x,1),[1,nSet,1,1]) 

        # cross set matching score
        score = self.cross_set_score(x)
        debug.append(score)

        # classificaiton of set matching
        fc2 = self.fc2(tf.reshape(score,[-1,1]))
        output2 = self.fc3(fc2)

        return output1, output2, debug

    # convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
    def cross_set_label(self, y):
        # rows of table
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])

        # cols of table       
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # if the class-labels are same, 1, otherwise 0
        return 1-tf.abs(y_rows - y_cols)

    def toBinaryLabel(self,y):
        dNum = tf.shape(y)[0]
        y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

        return tf.reshape(y,[dNum,-1])

    # train step
    def train_step(self,data):
        x, y_true = data
        
        with tf.GradientTape() as tape:
            
            # predict
            y_pred1, y_pred2, deb = self(x, training=True)

            if self.isPretrain:
                y_pred = y_pred1
            else:
                y_pred = y_pred2

                # convert to cross-set label
                y_true = self.cross_set_label(y_true)
                y_true = tf.reshape(y_true,-1)

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        
        # train using gradients
        trainable_vars = self.trainable_variables

        if self.isPretrain:
            trainable_vars = [v for v in trainable_vars if 'set' not in v.name] 
        else:
            trainable_vars = [v for v in trainable_vars if 'class' not in v.name]

        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        
        # predict
        y_pred1, y_pred2, _ = self(x, training=False)

        if self.isPretrain:
            y_pred = y_pred1
        else:
            y_pred = y_pred2

            # convert to cross-set label
            y_true = self.cross_set_label(y_true)
            y_true = tf.reshape(y_true,-1)

        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        x = data

        # predict
        y_pred1, y_pred2, _ = self(x, training=False)

        return y_pred1, y_pred2

# setting model
model = myModel(isPretrain=isPretrain, isSelfAttention=isSelfAttention, isCseft=isCseft)

# setting training, loss, metric to model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],run_eagerly=True)
#----------------------------

#----------------------------
# train
# make checkpoint callback to save trained parameters
checkpoint_path = "setmatching_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

if isTrain:

    if model.isPretrain:
        # # execute pretraining
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.2, callbacks=[cp_callback])

        model.isPretrain = False

    # execute training
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[cp_callback])
    #history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.2, callbacks=[cp_callback])

    # plot loss and accuracy
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
    # load trained parameters
    model.load_weights(checkpoint_path)
#----------------------------

# #----------------------------
# # evaluation with training data
# train_loss, train_acc = model.evaluate(x_train[:1000], y_train[:1000], verbose=0)
# print('Train data loss:', train_loss)
# print('Train data accuracy:', train_acc)
# #----------------------------

#----------------------------
# evaluation with validation data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_acc)
#----------------------------

y_pred1, y_pred2, deb = model(x_train[:4])

pdb.set_trace()