#IMPORTING ALL REQUIRED LIBS

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2 as cv2
from tqdm import tqdm

#DEFINE PATHS TO DATA SET
path="ImagePro"
files=os.listdir(path)
# LIST OF FILES IN PATH
# SORT PATH FROM A-Y
files.sort()

#print to see list
print(files)

#CREATE LIST OF IMAGES AND LABEL
image_array=[]
label_array=[]

#LOOP THROUGH EACH FILE IN FILES
for i in tqdm(range(len(files))):
    #LIST O IMAGES IN EACH FOLDER
    sub_file=os.listdir(path+"/"+files[i])
    #CHECKS LENGTH OF EACH FOLDER
    # PRINT(LEN(SUB_FILE)

    #LOOP THROUGH EACH FOLDER
    for j in range(len(sub_file)):
        #PATH OF EACH IMAGE
        #EXAMPLE:ImagePro/A/image_name1.jpg

        file_path=path+"/"+files[i]+"/"+sub_file[j]
        #READ EACH IMAGE

        image=cv2.imread(file_path)

        #RESIZE IMAGE BY 96x96
        image=cv2.resize(image,(96,96))
        #CONVERT BGR IMAGE TO RGB IMAGE
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        #ADD THIS IMAGE AT IMAGE_ARRAY
        image_array.append(image)

        #ADD label to label_array
        # i is the number from o to len(files)-1
        # so we can use it as label
        label_array.append(i)

#SAVE AND RUN


#CONVERT LIST TO ARRAY
image_array=np.array(image_array)
label_array=np.array(label_array,dtype="float")

#SPLIT THE DATASET INTO TEXT AND TRAIN THE SYSTEM
from sklearn.model_selection import train_test_split
#OUTPUT                                         TRAIN IMAGE     LABEL    SPLITING SIZE
X_train, X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)

del image_array,label_array

#TO FREE MEMORY
import gc
gc.collect()

#X_TRAIN WILL HAVE 85% OF IMAGES
#X_TEST WILL HAVE 15% OF IMAGES

#CREATE A MODEL
from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential,Model,load_model

model = Sequential()
#ADD PRETRAINED MODELS TO SEQUENTIAL MODEL
# IAM USING EFFICIENTNETB0 PRETRAINED MODEL (CAN USE A DIFFERENT MODEL)
pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(96,96,3),include_top=False)
model.add(pretrained_model)

#ADD POOLING TO MODEL
model.add(layers.GlobalAvgPool2D())

#ADD DROPOUT TO MODEL
#WE ADD DROPOUT TO INCREASE ACCURACY BY REDUCING OVERFITTING
model.add(layers.Dropout(0.3))

#FINALLY WE WILL ADD DENSE LAYER AS AN OUTPUT
model.add(layers.Dense(1))

#FOR SOME TENSORFLOW VERSION WE ARE REQUIRED TO BUIL MODEL
model.build(input_shape=(None,96,96,3))

#TO SEE MODEL SUMMERY
model.summary()

#SAVE AND RUN TO SEE MODEL SUMMARY
# wadanm kerenewa eth monawa wenewada kiyela dn na

#COMPILE MODEL
#YOU CAN USE DIFFERENT OPTIMIZER AND LOSS TO INCREASE ACCURACY
model.compile(optimizer="adam",loss="mae",metrics=["mae"])

#CREATE A CHECKPOINT TO SAVE BEST ACCURACY MODEL
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,monitor="val_mae",mode="auto",
                                                    save_best_only=True,save_weights_only=True)

# MONITOR: MONITOR VALIDATION "mae" loss to save MODEL
# MODE: USE TO SAVE MODEL WHEN "val_mae" IS MINIMUM OR MAXIMUM
# IT HAS 3 OPTIONS: "min","max","auto"
# FOR US YOU CAN SELECT EITHER "min" or "auto"
# WHEN "val_mae" REDUCE MODEL WILL BE SAVED
# "save_best_only": FALSE -> IT WILL SAVE ALL MODELS
# "save_weights_only:" SAVE ONLY WEIGHT

# CREATING LEARNING RATE REDUCER TO REDUCE LEARNING RATE WHEN ACCURACY DOES NOT IMPROVE
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,monitor="val_mae",mode="auto",cooldown=0,patience=5,
                                       verbose=1,min_lr=1e-6)

# FACTOR: WHEN IT IS REDUCE NEXT LEARN RATE WILL BE 0.9 TIMES OF CURRENT
# NEXT lr=0.9*current lr

# patience=X
# reduce lr after X epoch when accuracy does not improve
# verbose: show it after every epoch
# min_ir: minimum learning rate
# start training model

Epochs=100
Batch_Size=32
# SELECT BATCH SIZE ACCORDING TO YOUR GRAPHIC CARD
# X_train,X_test,Y_train,Y_test
history=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=Batch_Size,epochs=Epochs,
                  callbacks=[model_checkpoint,reduce_lr])

# BEFORE TRAINING YOU CAN DELETE image_array and label_array to increase RAM MEMORY

# AFTER THE TRAINING IS DONE LOAD BEST MODEL
model.load_weights(ckp_path)

# CONVERT MODEL TO TENSORFLOW LIGHT MODEL
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

# SAVE MODEL
with open("model.tflite","wb")as f:
    f.write(tflite_model)

#IF YOU WANT TO SEE PREDICTION RESULT ON TEST DATASET
prediction_val=model.predict(X_test,batch_size=32)

#PRINT FIRST 10 VALUES
print(prediction_val[:10])

#PRINT FIRST 10 VALUES OF Y_test
print(Y_test[:10])

# loss:0.4074 mae:0.4074 val_loss:0.3797 val_mae:0.3797
# WE HAVE mae and val_mae:
# mae: is on X_train
# val_mae: X_test
# If val_mae is reducing that means the model is improving