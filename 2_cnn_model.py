from tensorflow.keras.layers import Resizing,Input, Dropout, Dense,Conv2D,Conv2DTranspose,Flatten,Reshape,UpSampling2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.applications import VGG19, InceptionV3, MobileNetV2 ,ResNet50V2, DenseNet201, EfficientNetV2S, Xception
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

import tensorflow as tf
import matplotlib.pyplot as plt
import hickle as hkl
import argparse
import pickle
import numpy as np
import sklearn
#CONV

def simpleCNN(input_shape):
    
    input_img = Input(shape=input_shape)

    x = Conv2D(8, 3, activation='relu', name='conv1', input_shape=input_shape)(input_img)
    
    x = Conv2D(64, 3, activation='relu', name='conv2')(x)
        
    x = Conv2D(128, 3, activation='relu', name='conv3')(x)

    x = Flatten()(x)
    
    
    x = Dense(units=512, activation='relu',name='dense1')(x)
    
    x = Dense(units=128, activation='relu',name='dense2')(x)
    
    # output = Dense(units=10, activation='softmax',name='output')(x)
    
    return Model(inputs=input_img, outputs=x, name='AE')



def define_model(input_shape=(10, 12, 1), architecture='simpleCNN'):
    
    input_img = Input(shape=input_shape)

    x = Resizing(height=32,width=32, interpolation='nearest')(input_img)

    if architecture=='simpleCNN':
        model = simpleCNN(input_shape)
    elif architecture=='ResNet':
        model = ResNet50V2(
            include_top=False,
            weights=None,
            #weights='imagenet',
            input_tensor=input_img,
            #input_shape=(32,32,1),
            pooling=max,
            classes=9,
        )
        
    elif architecture=='VGG19':
        model = VGG19(
            include_top=False,
            weights=None,
            #weights='imagenet',
            input_tensor=x,
            #input_shape=(32,32,1),
            pooling=max,
            classes=9,
        )
    elif architecture=='MobileNet':
        model = MobileNetV2(
            include_top=False,
            weights=None,
            input_tensor=input_img,
            pooling=max,
        )
    elif architecture=='DenseNet':
        model = DenseNet201(
            include_top=False,
            weights=None,
            input_tensor=x,
            #input_shape=input_shape,
            pooling=max,
        )
    
    elif architecture=='EfficientNet':
        model = EfficientNetV2S(
            include_top=False,
            weights=None,
            input_tensor=input_img,
            pooling=max,
        )
       
    elif architecture=='Xception':
        model= Xception(
            include_top=False,
            weights=None,
            input_tensor=input_img,
            pooling=max,
        ) 
    
    elif architecture=='Inception':
        model= InceptionV3(
            include_top=False,
            weights=None,
            input_tensor=x,
            pooling=max,
        )
    #model = apply_regularization(model,1e-5,1e-4)

    # x = model.layers[-1].output
    print(model.layers[-1].output)
    x = Flatten()(model.layers[-1].output)
    
    output = Dense(units=9, activation='softmax',name='output')(x)
     
    model = Model(
            inputs=model.inputs,
            outputs=output)
    print(model.summary())
    return model
    

def main() -> None:
    
    # Load and compile Keras model
    parser = argparse.ArgumentParser(description="5G DDoS")
    
    parser.add_argument("--architecture",  required=True)
    parser.add_argument("--epochs",type=int,  required=True)
    parser.add_argument("--flow_length",type=int,  required=True)
  
    args = parser.parse_args()

    data = hkl.load('./data_'+str(args.flow_length)+'.hkl')

    X_train = data['xtrain']
    y_train = data['ytrain']
    '''
    new_X_train = []
    for i in X_train:
        new_X_train.append(np.pad(i, [(11, 11), (10, 10)], mode='constant'))
    X_train = np.array(new_X_train)
    '''
    X_test = data['xtest']
    y_test = data['ytest']
    '''
    new_X_test = []
    for i in X_test:
        new_X_test.append(np.pad(i, [(11, 11), (10, 10)], mode='constant'))
    X_test = np.array(new_X_test)
    '''

    batch_size = 256
    epochs = 30

    print("Y train", np.unique(y_train,return_counts=True))
    print("Y test", np.unique(y_test,return_counts=True))
    

    
    #model = Conv2D_1()
    
    model = define_model(input_shape=(args.flow_length,12,1),architecture=args.architecture)
    

    print(model.summary())
    
    
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(y_train),
                                                 y = y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    
    checkpoint_filepath = '/tmp/checkpoint'

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=True)
        
    opt = Adam(learning_rate=0.001)#learning_rate=0.001)
    
    model.compile(optimizer= opt, loss='sparse_categorical_crossentropy',metrics='accuracy')
    
    history = model.fit(
            X_train,
            y_train,
            batch_size,
            args.epochs,
            validation_data=(X_test,y_test),
            #validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=[model_checkpoint_callback]
        )
    
   
    model.load_weights(checkpoint_filepath)

    
    with open('file_results_'+str(args.flow_length)+'/'+args.architecture+'historyDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
   
    model.save('file_results_'+str(args.flow_length)+'/'+args.architecture+".h5")
     
    pred = model.predict(X_test)
    test_predictions = np.argmax(pred, axis=1)
    conf_mat = confusion_matrix(y_test, test_predictions)

    cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Benign','Attack'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cmn)
        
    disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.show()

    f1_score = np.round(sklearn.metrics.f1_score(y_test, test_predictions,average="weighted"), 5)
    acc = np.round(sklearn.metrics.accuracy_score(y_test, test_predictions), 5)
    precision = np.round(sklearn.metrics.precision_score(y_test, test_predictions,average="weighted"), 5)
    recall = np.round(sklearn.metrics.recall_score(y_test, test_predictions,average="weighted"), 5)

    print('Acc = %.5f, f1_score = %.5f, precision = %.5f, recall = %.5f' % (acc,f1_score,precision,recall))
    
    
    
    
if __name__=="__main__":
    main()
