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
# set parameters
isTrain = True

# flag of self-attention and cseft
isAttention = True

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

# #----------------------------
# # multi-head CS function to make cros-set matching score map
# class cross_set_score_old(tf.keras.layers.Layer):
#     def __init__(self, head_size=20, num_heads=2):
#         super(cross_set_score_old, self).__init__()
#         self.head_size = head_size
#         self.num_heads = num_heads

#         # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
#         # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
#         self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,activation='relu',use_bias=False)

#         # linear function to combine multi-head score maps
#         self.conv = tf.keras.layers.Conv2D(filters=1, strides=(1,1), padding='same', kernel_size=(1,1))

#     def call(self, x):

#         # linear transofrmation (nSet*nItem,head_size*num_heads)
#         x = self.linear(x)

#         # reshape (nSet*nItem,head_size*num_heads) to (num_heads, nSet*nItem,head_size)
#         x = tf.transpose(tf.reshape(x,[-1,self.num_heads,self.head_size]),[1,0,2])

#         # inner products between all pairs of items, outputing (num_heads, nSet*nItem, nSet*nItem)-score map
#         xx = tf.matmul(x,tf.transpose(x,[0,2,1]))/tf.sqrt(tf.cast(self.head_size,tf.float32))

#         # reshape (num_heads, nSet*nItem, nSet*nItem,1)
#         xx = tf.expand_dims(xx,-1)

#         # sum up score-map in each block of size (nItem, nItem) using conv2d with weights of ones
#         # outputing (num_heads, nSet, nSet, 1)-score map
#         scores = tf.keras.layers.Conv2D(filters=1,strides=(nItem,nItem),kernel_size=(nItem,nItem),
#                     trainable=False,use_bias=False,weights=[tf.ones((nItem,nItem,1,1))])(xx)

#         # devided by the two numbers of items of two sets（NOTE that nItem is SIMPLY fixed）
#         scores = scores/nItem/nItem

#         # linearly combine multi-head score maps
#         # reshape (num_heads, nSet, nSet, 1) to (1, nSet, nSet, num_heads)
#         scores = tf.transpose(scores,[3,1,2,0])
#         scores = self.conv(scores)

#         return scores[0,:,:,0]
# #----------------------------

#----------------------------
# multi-head CS function to make cros-set matching score map
class cross_set_score(tf.keras.layers.Layer):
    def __init__(self, head_size=20, num_heads=2):
        super(cross_set_score, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        self.linear = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(),use_bias=False)

        # linear function to combine multi-head score maps
        self.conv = tf.keras.layers.Conv2D(filters=1, strides=(1,1), padding='same', kernel_size=(1,1),use_bias=False)

    def call(self, x):
        nSet = tf.shape(x)[1]
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        # linear transofrmation from (nSet*nItem, nSet, Xdim) to (nSet*nItem, nSet, head_size*num_heads)
        x = self.linear(x)

        # reshape (nSet*nItem,head_size*num_heads) to (num_heads, nSet*nItem,head_size)
        x = tf.transpose(tf.reshape(tf.expand_dims(x,2),[nSet*nItem, nSet, self.num_heads, self.head_size]),[2,0,1,3])        

        # compute inner products between all pairs of items with cross-set feature (cseft)
        # Between set #1 and set #2, cseft x[:,0:2,1] and x[:,2:4,0] are extracted to compute inner product when nItem=2
        # More generally, between set #i and set #j, cseft x[:,j*2:j*2+2,i] and x[:,i*2:i*2+2,j] are extracted.
        # Outputing (nSet, nSet, num_heads)-score map
        scores = tf.stack([[tf.reduce_sum(tf.reduce_sum(tf.matmul(x[:,j*nItem:j*nItem+nItem,i],tf.transpose(x[:,i*nItem:i*nItem+nItem,j],[0,2,1]))/sqrt_head_size,axis=1),axis=1)
                         for i in range(nSet)] for j in range(nSet)])

        # devided by the two numbers of items of two sets（NOTE that nItem is SIMPLY fixed）
        scores = scores/nItem/nItem

        # linearly combine multi-head score maps
        # reshape (nSet, nSet, num_heads) to (1, nSet, nSet, num_heads)
        scores = tf.expand_dims(scores,0)
        scores = self.conv(scores)

        return scores[0,:,:,0]
#----------------------------        

#----------------------------
# cross-set feature (cseft)
class cseft(tf.keras.layers.Layer):
    def __init__(self, isOnlySelf=False, head_size=20, num_heads=2):
        super(cseft, self).__init__()
        self.isOnlySelf = isOnlySelf
        self.head_size = head_size
        self.num_heads = num_heads        

        # multi-head linear function, l(x|W_0), l(x|W_1)...l(x|W_num_heads) for each item feature vector x.
        # one big linear function with weights of W_0, W_1, ..., W_num_heads outputs head_size*num_heads-dim vector
        # self.linear1 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,activation='relu',use_bias=False)
        # self.linear2 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,activation='relu',use_bias=False)
        # self.linear3 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,activation='relu',use_bias=False)
        self.linear1 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(), use_bias=False)
        self.linear2 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,kernel_constraint=tf.keras.constraints.NonNeg(), use_bias=False)
        self.linear3 = tf.keras.layers.Dense(units=self.head_size*self.num_heads,use_bias=False)

        # linear function to combine multi-head score maps
        self.conv = tf.keras.layers.Conv2D(filters=1, strides=(1,1), padding='same', kernel_size=(1,1),use_bias=False)

    def call(self, x, y):
        # number of sets
        nSet = tf.cast(tf.shape(x)[0]/nItem,tf.int32)
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size,tf.float32))

        # linear transofrmation (nSet*nItem, head_size*num_heads)
        x = self.linear1(x)
        y1 = self.linear2(y)
        y2 = self.linear3(y)

        # reshape (nSet*nItem, head_size*num_heads) to (num_heads, nSet*nItem, head_size)
        x = tf.transpose(tf.reshape(x,[-1,self.num_heads,self.head_size]),[1,0,2])
        y1 = tf.transpose(tf.reshape(y1,[-1,self.num_heads,self.head_size]),[1,0,2])
        y2 = tf.transpose(tf.reshape(y2,[-1,self.num_heads,self.head_size]),[1,0,2])

        # inner products between all pairs of items, outputing (num_heads, nSet*nItem, nSet*nItem)-score map        
        xy1 = tf.matmul(x,tf.transpose(y1,[0,2,1]))/sqrt_head_size

        # block diagonal matrix containing the block (num_heads, nItem, head_size) of y2 in diagonal elements
        # outputs (num_heads, nSet*nItem, nSet*head_size)
        zeros = tf.zeros((self.num_heads, nItem, self.head_size))
        y2_block_diag = tf.concat([tf.concat([tf.tile(zeros,[1,1,i]),y2[:,i*nItem:(i+1)*nItem],tf.tile(zeros,[1,1,nSet-(i+1)])],axis=2) 
                        for i in range(nSet)],axis=1) 

        # computing weighted y2 by xy1, outputing (num_heads, nSet*nItem, nSet*head_size)
        newx = tf.matmul(xy1, y2_block_diag)

        # linearly combine multi-head score maps
        # reshape (num_heads, nSet*nItem, nSet*head_size) to (1, nSet*nItem, nSet*head_size, num_heads)
        newx = tf.expand_dims(newx,-1)
        newx = tf.transpose(newx,[3,1,2,0])
        newx = newx/nItem

        # combine heads, outputing (1, nSet*nItem, nSet*head_size, 1)
        newx = self.conv(newx)

        # reshape (1, nSet*nItem, nSet*head_size, 1) to (nSet*nItem, nSet*head_size)
        newx = newx[0,:,:,0]

        if self.isOnlySelf:
            # extract only diagonal block (nItem, head_size), outputing (nSet*nItem, head_size)
            newx = tf.concat([newx[i*nItem:(i+1)*nItem,i*self.head_size:(i+1)*self.head_size] for i in range(nSet)],axis=0) 
        else:
            # reshape (nSet*nItem, nSet*head_size) to (nSet*nItem, nSet, head_size)
            newx = tf.reshape(tf.expand_dims(newx,1),[-1,nSet,self.head_size])
        

        return newx
#----------------------------        

#----------------------------
# design network architecture
class myModel(tf.keras.Model):
    def __init__(self, isAttention=True, num_heads=2):
        super(myModel, self).__init__()
        self.isAttention = isAttention

        # conv, fc, defc & deconv layers
        self.conv1 = tf.keras.layers.Conv2D(filters=baseChn, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=baseChn*2, strides=(2,2), padding='same', kernel_size=(3,3), activation='relu')
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.cross_set_score = cross_set_score(head_size=baseChn*2,num_heads=num_heads)
        self.cseft_enc = cseft(isOnlySelf=True, head_size=baseChn*2,num_heads=num_heads)
        self.cseft_dec = cseft(head_size=baseChn*2,num_heads=num_heads)
        self.fc = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, x):

        # reshape (nSet, nItem, H, W, C) to (nSet*nItem, H, W, C)
        nSet = tf.shape(x)[0]
        x = tf.reshape(x,[-1,H,W,C])
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.globalpool(x)
        cnnfet = x

        # Attention
        if self.isAttention:
            # encoder (self-set attention)
            # input: (nSet*nItem, D)
            # output: (nSet*nItem, D)
            x = self.cseft_enc(x,x)
            cseft_enc = x

            # decoder (cross-set transofrmation:cseft)
            # input: (nSet*nItem, D)
            # output: (nSet*nItem, nSet, D)
            x = self.cseft_dec(x,x)
            cseft_dec = x

        else:
            x = tf.tile(tf.expand_dims(x,1),[1,nSet,1]) 

        # cross set matching score
        score = self.cross_set_score(x)

        # classificaiton
        output = self.fc(tf.reshape(score,[-1,1]))

        return output, (score, cnnfet, cseft_enc, cseft_dec)

    # convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
    def cross_set_label(self, y):
        # rows of table
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])

        # cols of table       
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # if the class-labels are same, 1, otherwise 0
        return 1-tf.abs(y_rows - y_cols)

    # def toBinaryLabel(self,y):
    #     dNum = tf.shape(y)[0]
    #     y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

    #     return tf.reshape(y,[dNum,-1])

    # train step
    def train_step(self,data):
        x, y_true = data
        
        with tf.GradientTape() as tape:
            
            # predict
            y_pred, deb = self(x, training=True)

            # convert to cross-set label
            y_true = self.cross_set_label(y_true)
            y_true = tf.reshape(y_true,-1)

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        
        # train using gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        #pdb.set_trace()

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        
        # predict
        y_pred, _ = self(x, training=False)

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
        y_pred, _ = self(x, training=False)

        return y_pred

# setting model
model = myModel(isAttention=isAttention)

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

    # execute training
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[cp_callback])

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

#----------------------------
# evaluation with training data
train_loss, train_acc = model.evaluate(x_train[:1000], y_train[:1000], verbose=0)
print('Train data loss:', train_loss)
print('Train data accuracy:', train_acc)
#----------------------------

#----------------------------
# evaluation with validation data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test data loss:', test_loss)
print('Test data accuracy:', test_acc)
#----------------------------

y_pred,score=model(x_train[:4])

pdb.set_trace()