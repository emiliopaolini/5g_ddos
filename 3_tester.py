import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

from tensorflow.keras.layers import Input, Dense,Conv2D,Conv2DTranspose,Flatten,Reshape,UpSampling2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import argparse
import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import hickle as hkl
import pickle
import time
import numpy as np
import sklearn
import larq as lq
import seaborn as sns
#CONV

#https://deci.ai/blog/measure-inference-time-deep-neural-networks/

def Conv2D_1(input_shape=(10, 12, 1), filters=[8, 16, 32, 16]):
    
    input_img = Input(shape=input_shape)

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(8, 3, activation='relu', name='conv1', input_shape=input_shape)(input_img)

    x = Conv2D(64, 3, activation='relu', name='conv2')(x)
    
    #x = AveragePooling2D((2,2))(x)
    
    x = Conv2D(128, 3, activation='relu', name='conv3')(x)
    
    #x = MaxPooling2D((2,2))(x)

    #x = Conv2D(16, 3, activation='relu', name='conv3')(x)

    #x = Conv2D(8, 3, activation='relu', name='conv4')(x)


    x = Flatten()(x)
    x = Dense(units=512, activation='relu',name='dense1')(x)
    
    x = Dense(units=128, activation='relu',name='dense2')(x)
    output = Dense(units=10, activation='softmax',name='output')(x)
    '''
    encoded = Dense(units=filters[3], name='embedding')(x)

    x = Dense(units=24192, activation='relu',name='dense_dec')(encoded)

    x = Reshape((42,3,192))(x)
    x = Conv2DTranspose(64, 3, activation='relu', name='deconv5')(x)

    x = Conv2DTranspose(32, 3, activation='relu', name='deconv4')(x)

    x = Conv2DTranspose(filters[2], 3, padding=pad3, activation='relu', name='deconv3')(x)
    
    x = Conv2DTranspose(filters[1], 3, activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(input_shape[2], 3, padding='same', name='deconv1')(x)
    '''
    return Model(inputs=input_img, outputs=output, name='AE')

def main() -> None:
    
    # Load and compile Keras model
    parser = argparse.ArgumentParser(description="5G DDoS")
    
    parser.add_argument("--architecture",  required=True)
  
    parser.add_argument("--flow_length", type=int, default=10, required=True)
  
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
    batch_size = 128
    epochs = 30

    print("Y train", np.unique(y_train,return_counts=True))
    print("Y test", np.unique(y_test,return_counts=True))
    
    
    with open('file_results_'+str(args.flow_length)+'/'+args.architecture+'historyDict', 'rb') as fp:
        history  = pickle.load(fp)
    
    print(history)
    model = load_model('file_results_'+str(args.flow_length)+'/'+args.architecture+".h5")
     
    print(lq.models.summary(model))
    print(model.summary())
    
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('pictures_'+str(args.flow_length)+'/'+args.architecture+'_accuracy.png')
    plt.clf()


    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('pictures_'+str(args.flow_length)+'/'+args.architecture+'_loss.png')
    plt.clf()

    
    start = time.perf_counter()

    pred = model.predict(X_test)
    
    end = time.perf_counter()

    tm = end-start
    print("Prediction time:",tm)
    print("test length: ",X_test.shape)
    test_predictions = np.argmax(pred, axis=1)
    conf_mat = confusion_matrix(y_test, test_predictions)

    cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Benign','Attack'])
    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    #disp = disp.plot(include_values=True, cmap='plasma', fmt='.2%', ax=None, xticks_rotation='horizontal')

    
    sns.heatmap(cmn, annot=True, fmt='.1%', cmap='viridis')
    plt.show()
    plt.savefig('pictures_'+str(args.flow_length)+'/'+args.architecture+'_conf_matrix.png')
    plt.clf()

    f1_score = np.round(sklearn.metrics.f1_score(y_test, test_predictions,average="weighted"), 5)
    acc = np.round(sklearn.metrics.accuracy_score(y_test, test_predictions), 5)
    precision = np.round(sklearn.metrics.precision_score(y_test, test_predictions,average="weighted"), 5)
    recall = np.round(sklearn.metrics.recall_score(y_test, test_predictions,average="weighted"), 5)

    print('Acc = %.5f, f1_score = %.5f, precision = %.5f, recall = %.5f' % (acc,f1_score,precision,recall))
    
    
    
    
if __name__=="__main__":
    main()
