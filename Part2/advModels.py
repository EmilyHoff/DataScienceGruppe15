from multiprocessing.dummy import active_children
import os
import shutil
from tkinter import E
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import sys
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
# import fasttext
#sys.path.insert(0,"../")

#import main
import wandb

#df = None
'''
def dense32AvgPool(firstAc=None,secondAc=None,thirdAc=None):
    input = tf.keras.layers.Input(shape=(),dtype=tf.string,name="Input layer")

    preprocessing = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",name="Preprocessing") #Evt kig på denne, da den truncater artikler til mindre størrelse

    preprocessedInput = preprocessing(input)

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",name="Encoder",trainable=True)

    encodedInput = encoder(preprocessedInput)
    output = encodedInput["pooled_output"]

    dense = tf.keras.layers.Dense(32,activation=firstAc,name="1")(output)
    dense = tf.reshape(dense, [tf.shape(dense)[0], 1, tf.shape(dense)[1]])
    lstm = tf.keras.layers.LSTM(16,secondAc)(dense)
    lstm = tf.reshape(lstm, [tf.shape(lstm)[0], tf.shape(lstm)[2]])
    drop = tf.keras.layers.GaussianDropout(rate=0.1)(lstm)
    dense = tf.keras.layers.Dense(8,activation=thirdAc,name="3")(drop)
    output = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid,name="4")(dense)

    return tf.keras.Model(input,output)


def simple(firstAc=None,secondAc=None,thirdAc=None):
    input = tf.keras.layers.Input(shape=(),dtype=tf.string,name="Input layer")

    preprocessing = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",name="Preprocessing") #Evt kig på denne, da den truncater artikler til mindre størrelse

    preprocessedInput = preprocessing(input)

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",name="Encoder",trainable=True)

    encodedInput = encoder(preprocessedInput)
    output = encodedInput["pooled_output"]

    dense = tf.keras.layers.Dense(16,activation=firstAc,name="1")(output)
    drop = tf.keras.layers.Dropout(0.1)(dense)
    dense = tf.keras.layers.Dense(8,activation=secondAc,name="2")(drop)
    output = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid,name="3")(dense)

    return tf.keras.Model(input,output)


modelLi = [dense32AvgPool,simple]
modelAc = [tf.keras.activations.tanh,tf.keras.activations.sigmoid,tf.keras.activations.relu]
sweep_configuration = {
    "method":"bayes",
    "metric":{"name":"accuracy","goal":"maximize"},
    "parameters":{
        "firstAc":{"values":[0,1,2]},
        "secondAc":{"values":[0,1,2]},
        "modelNumber":{"values":[0,1]},
        "lr":{"values":[0.01,1,0.0001]}
    }
}

def bert(config):
    global df
    trainX = tf.convert_to_tensor(df["content"][:3*(len(df)//4)-1].values,dtype=tf.string)
    trainY = tf.convert_to_tensor(df["type"][:3*(len(df)//4)-1].astype(np.float32),dtype=tf.float32)

    testX = tf.convert_to_tensor(df["content"][3*(len(df)//4)+1:].values,dtype=tf.string)
    testY = tf.convert_to_tensor(df["type"][3*(len(df)//4)+1:].astype(np.float32),dtype=tf.float32)

    batch_size = 32

    # Split data into batches
    num_batches = len(trainX) // batch_size
    X_batches = np.array_split(trainX, num_batches)
    y_batches = np.array_split(trainY, num_batches)

    bertModel = modelLi[config.modelNumber](modelAc[config.firstAc],modelAc[config.secondAc],modelAc[config.firstAc])

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    epochs = 5
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    bertModel.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])

    for i in range(num_batches):
        xBatch = X_batches[i]
        yBatch = y_batches[i]
        bertModel.train_on_batch(xBatch,yBatch)

    #bertModel.fit(trainX,trainY,epochs=5)
    loss, accuracy = bertModel.evaluate(testX,testY)
    return loss,accuracy

def wandbHandle():
    wandb.init(project='fake-news-bert-tuning')
    loss, accuracy = bert(wandb.config)
    wandb.log({
        'accuracy':accuracy,
        'loss':loss
    })


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='fake-news-bert-tuning')
    df = main.main()
    df = df[:1000]
    wandb.agent(sweep_id, function=wandbHandle, count=4)




def bert(df):
    trainX = tf.convert_to_tensor(df["content"][:3*(len(df)//4)-1].values,dtype=tf.string)
    trainY = tf.convert_to_tensor(df["type"][:3*(len(df)//4)-1].astype(np.float32),dtype=tf.float32)

    print(f"This is the train data: {trainX}")
    print(f"This is the type: {trainY}")

    testX = tf.convert_to_tensor(df["content"][3*(len(df)//4)+1:].values,dtype=tf.string)
    testY = tf.convert_to_tensor(df["type"][3*(len(df)//4)+1:].astype(np.float32),dtype=tf.float32)

    batch_size = 32

    # Split data into batches
    num_batches = len(trainX) // batch_size
    X_batches = np.array_split(trainX, num_batches)
    y_batches = np.array_split(trainY, num_batches)

    def simple():
        input = tf.keras.layers.Input(shape=(),dtype=tf.string,name="Input layer")

        preprocessing = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",name="Preprocessing") #Evt kig på denne, da den truncater artikler til mindre størrelse
        preprocessedInput = preprocessing(input)

        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2",name="Encoder",trainable=True)
        encodedInput = encoder(preprocessedInput)
        output = encodedInput["pooled_output"]

        #dense = tf.keras.layers.Dense(32,activation=tf.keras.activations.tanh,name="1")(output)
        #drop = tf.keras.layers.GaussianDropout(rate=0.1)(dense)
        dense = tf.keras.layers.Dense(16,activation=tf.keras.activations.relu,name="2")(output)
        drop = tf.keras.layers.GaussianDropout(rate=0.1)(dense)
        dense = tf.keras.layers.Dense(8,activation=tf.keras.activations.relu,name="3")(drop)
        output = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid,name="4")(dense)

        return tf.keras.Model(input,output)

    bertModel = simple()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.4)

    bertModel.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])

    for i in range(num_batches):
        print(f"batch: {i}")
        xBatch = X_batches[i]
        yBatch = y_batches[i]
        bertModel.train_on_batch(xBatch,yBatch)

    #bertModel.fit(trainX,trainY,epochs=5)
    loss, accuracy = bertModel.evaluate(testX,testY)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    return loss,accuracy

'''
from transformers import pipeline
import os

def bert(encoded,embedding_matrix,labels,vocab_size,split):
    print(f"split is {split},length of encoded: {len(encoded)}, length of labels {len(labels)}")

    train,labTrain = encoded[:split],labels[:split]
    test,labTest = encoded[split:],labels[split:]

    print(f"labTrain {len(labTrain)}")
    print(f"labTest {len(labTest)}")

    import random
    shuffler = list(zip(train,labTrain))

    random.shuffle(shuffler)

    feat,lab = [],[]

    for fea,la in shuffler:
        feat.append(fea)
        lab.append(la)

    trainX = np.array(feat)
    trainY = np.array(lab)

    batch_size = 32
    num_batches = len(trainX) // batch_size
    X_batches = np.array_split(trainX, num_batches)
    y_batches = np.array_split(trainY, num_batches)

    bertModel = Sequential()

    bertModel.add(tf.keras.layers.Embedding(vocab_size+1,150,weights=[embedding_matrix]))
    bertModel.add(tf.keras.layers.Dense(64))
    bertModel.add(tf.keras.layers.Dropout(0.2))
    bertModel.add(tf.keras.layers.LSTM(32))
    bertModel.add(tf.keras.layers.Dropout(0.3))
    bertModel.add(tf.keras.layers.Dense(16))
    bertModel.add(tf.keras.layers.Dense(1))

    bertModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

    for i in range(num_batches):
        print(f"batch: {i}")
        xBatch = X_batches[i]
        yBatch = y_batches[i]
        bertModel.train_on_batch(xBatch,yBatch)
    print(f"This is the test: {test}")
    print(f"This is the labels: {labTest}")
    loss, accuracy = bertModel.evaluate(np.array(test),np.array(labTest))
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    return loss,accuracy


def fastClass(df,split):

    train,trainLabels = df["content"][:split],df["type"][:split]
    test,testLabels = df["content"][split:],df["type"][split:]

    with open("fastTextTrain.txt","w") as f:
        for con,lab in zip(train["content"].tolist(),trainLabels["type"].tolist()):
            f.write(f"__label__{lab} {con}")

    with open("fastTextTest.txt","w") as f:
        for con,lab in zip(test["content"].tolist(),testLabels["type"].tolist()):
            f.write(f"__label__{lab} {con}")

    model = fasttext.train_supervised("fastTextWords.txt",
                                      lr=0.5,
                                      epoch=25,
                                      wordNgrams=2,
                                      bucket=200000,
                                      dim=100,
                                      loss='hs',
                                      ws=5)
    print(f"Test score: {model.test('fastTextTest.txt',k=-1)}")

from sklearn.cluster import KMeans

def kmeans(df,split):
    train,labTrain = df["content"][:split],df["type"][:split]
    test,labTest = df["content"][split:],df["type"][split:]

    train = np.array(train)
    test = np.array(test)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(train)

    y_pred = kmeans.predict(test)
    print(f"Accuracy score: {accuracy_score(y_pred,labTest)}")


def svm(df,split):

    clf = svm.SVC()

    clf.fit(X[:3*(len(df)//4)-1],y[:3*(len(df)//4)-1])

    y_pred = clf.predict(X[3*(len(df)//4)+1:])

    return accuracy_score(y_pred,y[3*(len(df)//4)+1:])

