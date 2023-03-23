from multiprocessing.dummy import active_children
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
#import tensorflow_text as text
import numpy as np
#from official.nlp import optimization




def bert(df):
    trainX = tf.convert_to_tensor(df["content"][:3*(len(df)//4)-1].values,dtype=tf.string)
    trainY = tf.convert_to_tensor(df["type"][:3*(len(df)//4)-1].astype(np.float32),dtype=tf.float32)
    
    testX = tf.convert_to_tensor(df["content"][3*(len(df)//4)+1:].values,dtype=tf.string)
    testY = tf.convert_to_tensor(df["type"][3*(len(df)//4)+1:].astype(np.float32),dtype=tf.float32)
    
    #prePro = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    #bertModel = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")
    #print(bertModel(prePro([x for x in df["content"]])))
    
    batch_size = 32

    # Split data into batches
    num_batches = len(trainX) // batch_size
    X_batches = np.array_split(trainX, num_batches)
    y_batches = np.array_split(trainY, num_batches)
    
    
    def model():
        input = tf.keras.layers.Input(shape=(),dtype=tf.string,name="Input layer")
        
        preprocessing = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",name="Preprocessing") #Evt kig på denne, da den truncater artikler til mindre størrelse
        
        preprocessedInput = preprocessing(input)
        
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",name="Encoder",trainable=True)
        
        encodedInput = encoder(preprocessedInput)
        output = encodedInput["pooled_output"]
        
        drop = tf.keras.layers.Dropout(0.1)(output)
        
        dense = tf.keras.layers.Dense(16,activation=tf.keras.activations.tanh,name="1")(drop)
        drop = tf.keras.layers.Dropout(0.1)(dense)
        dense = tf.keras.layers.Dense(8,activation=tf.keras.activations.tanh,name="2")(drop)
        drop = tf.keras.layers.Dropout(0.1)(dense)
        output = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid,name="3")(drop)
        
        return tf.keras.Model(input,output)
    
    bertModel = model()
    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    
    epochs = 5    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    bertModel.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])
    
    for i in range(num_batches):
        xBatch = X_batches[i]
        yBatch = y_batches[i]
        bertModel.train_on_batch(xBatch,yBatch)
    
    #bertModel.fit(trainX,trainY,epochs=5)
    loss, accuracy = bertModel.evaluate(testX,testY)
    
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    
    