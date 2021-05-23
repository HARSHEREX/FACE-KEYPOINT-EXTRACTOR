########################################################################################################
####
#### if running in spyder please set console working dir before running the program 
####
########################################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, History
from plot_keras_history import plot_history


##dir setup
os.system('cls')
cwd = os.getcwd()
source_dir = '\\'.join(cwd.split('\\')[:-1])


# training data loading
def load(filename):
        #reading training data from csv
        training = pd.read_csv(filename)
        # combining all imagesm of 96*96 to a stack and normalize images and then reshape them to 4d array for cnn input layer
        xtr = np.vstack(training.Image.apply(lambda x: np.array(np.array(x.split()).astype('uint8'))/255).values).reshape(-1,96,96,1)
        # x y coordinates of 15 key point 
        ytr = training.iloc[:,:-1].values
        # column names for result df
        cols =  training.iloc[:,:-1].columns
        # x = (x-MeanOfx)/Mean value of coordinate to normalize them 
        ymean = ytr.mean()
        ytr = (ytr - ymean)/ymean
        # return X_traing_set , y_Training_set , column names for 15 keypoint df
        return xtr,ytr,cols

# train function 
def train():
    # change dir to data folder 
    os.chdir(source_dir+'/Resource/data/')
    # read training file
    filename = 'training/training.csv'
    print('loading_data')
    # call load function
    X_train,y_train,cols = load(filename)
    
    ### model architecture
    def forge_model():
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=X_train[0].shape)) # Input shape: (96, 96, 1)
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        # Convert all values to 1D array
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(30))
        return model
    
    # hyperparameter
    epochs = 45
    batch_size = 64
    # geting model architecture and history object
    model = forge_model()
    hist = History()
    os.chdir(source_dir+'/Resource')
    # saving mopdel checkpoint based on improvement in validation loss
    checkpointer = ModelCheckpoint(filepath='model/checkpoint_model.hdf5', verbose=1, save_best_only=True)
    # Complie and fit Model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, hist], verbose=1)
    # ploting model performance
    plot_history(hist)
    plt.show()
    #saving model performance to /Resource/model
    plot_history(hist, path="model/performance graph.jpg")
    #saving model to Resource/model
    model.save('model/kpmodel.h5')
    # returning trained model 
    return  model 





# get result on custom data set
def confirm():
    #interface for the  program flow
    train_ = input('\n(Train model) press "Y" | "N" : ')
    if train_ == 'Y' or train_ == 'y':
        model = train()


    #interface for the  program flow
    test_ = input('\nWant to extract facial points from image dataset "Y" | "N" ')

    if test_=='Y' or test_=='y':

        # trying to load model 
        try :
            os.chdir(source_dir+'/Resource/model')
            print('loading_model')
            model = load_model('kpmodel.h5')
            # exceptions :  console running directory is different than program's path (both must be same)
        except Exception as e:
            # handling exception 
            print(e,'\n\n>>>>>>>>>>>> MODEL NOT FOUND MUST HAVE TO TRAIN <<<<<<<<<<<<<<<\n >>>>> or Set the console working dir to the python program dir \n \t on GTX 1650 4 GB training takes 4.5 minuts ')
            confirm()


        # result visualization and keypoint file generation function
        def result_visual(res,names):
            ymean =  47.5856
            cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
            'right_eye_center_y', 'left_eye_inner_corner_x',
            'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
            'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
            'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
            'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
            'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
            'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
            'mouth_right_corner_y', 'mouth_center_top_lip_x',
            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y']
            # rescaling coordinates to original coordinates
            res = pd.DataFrame(res*ymean+ymean,columns=cols)
            # saving file dir
            os.chdir(source_dir+'/resource/keypoint_csv/')
            res['name']=names
            # asking file name for checkpoint file
            fn = input('please enter keypoint file name [ex :- file.csv] : ')
            res.to_csv(fn,index=False)
            print('\n\t file saved in FACE-KEYPOINT-EXTRACTOR\Resource\keypoint_csv\\',fn)
            # making mask image of coordinates
            resx = res.iloc[:,list(range(0,30,2))].values.round().astype(int)
            resy = res.iloc[:,list(range(1,30,2))].values.round().astype(int)
            maps = []
            reslen = len(resx[0])
            kernel = np.ones((2,2), np.uint8)
            for i in range(len(res)):
                loc_map = np.zeros((96,96))
                for j in range(reslen):
                    loc_map[resy[i][j],resx[i][j]]=1
                loc_map = cv2.dilate(loc_map,kernel,iterations=1)
                loc_map = 1- loc_map
                maps.append(loc_map)
            return maps
        
        # data loading and processing from custom dir
        custom_data_dir = input('"\n\n\n\nimages must be of croped face only else you will not get useable results\n\nenter images data DIR" or press "Ctrl+c" to stop the process: ')
        image_names = os.listdir(custom_data_dir)
        os.chdir(custom_data_dir)
        read_img = lambda x: cv2.imread(x)
        cvtclr = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        resize = lambda x : cv2.resize(x, (96, 96))
        imgs = pd.DataFrame([[i,resize(cvtclr(read_img(i)))] for i in image_names],columns=['name','image'])
        x = np.vstack(imgs.image.apply(lambda x: x.astype('uint8')/255).values).reshape(-1,96,96,1)
        result_custom_real = model.predict(x)
        names = imgs.name.tolist()
        map_ = result_visual(result_custom_real,names)

        # saving result image to specified directory
        def test_save(imgs,maps,save_dir):
            names = imgs.name.values
            imgs=imgs.image.values
            os.chdir(save_dir)
            for n,i,m in zip(names,imgs,maps):
                i[i==0]=1
                x  = i*m
                x[x== 0] = 255
                plt.imsave(str(n)+"_result_"+'.jpg',x,cmap='gray')
        
        save_dir = input('\n\nenter result image save dir or just press enter : ')
        if save_dir:
            test_save(imgs,map_,save_dir)
            print(' >>>>>>>>>> result images saved ')
            confirm()
    else:
        exit()

confirm()



