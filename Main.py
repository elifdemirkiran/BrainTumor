from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from ui_UInterface import *
import sys,os,glob
import numpy as np
import pandas as pd
import seaborn as sns
from keras.utils import img_to_array, img_to_array, load_img
from tqdm import tqdm
import skimage
from skimage.transform import resize
import sys,os,glob, cv2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB5
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import layers
from keras.models import Sequential
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report,roc_auc_score, roc_curve,confusion_matrix,accuracy_score,auc

import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils import shuffle as shf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from PyQt5.QtGui import QPixmap
class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menu_New = QMenu()
        self.train=""
        self.test=""
        self.test_data=[]
        self.test_labels = []
        self.file=[0]
        self.h5file=[0]
        self.fname=[0]
        self.ui.btnTespit.clicked.connect(self.ResimTespiti) 
        self.ui.buttonCM.clicked.connect(self.Matrix) 
        self.ui.buttonLoss.clicked.connect(self.KayipMatrisi) 
        self.ui.buttonAccuracy.clicked.connect(self.DogrulukMatrisi) 
        self.ui.Veriseti_Yukle.clicked.connect(self.KlasorAc)  
        self.ui.buttonVeriCokla.clicked.connect(self.VeriCoklama)  
        self.ui.ResimSec.clicked.connect(self.ResimSec)
        self.ui.h5Sec.clicked.connect(self.h5Sec)
        self.ui.buttonModelEgit.clicked.connect(self.modelEgit)

        self.ui.buttonSinifCalistir.clicked.connect(self.Siniflandir)
    def Matrix(self):
        if self.h5file[0]!=0: 
            code = {'glioma':0 ,'meningioma':1,'notumor':2}
            X_test = []
            y_test = []
            for folder in  os.listdir(self.test) : 
                files = glob.glob(pathname= str(self.test + folder + '/*.jpg'))
                for file in files: 
                    image = cv2.imread(file)
                    image_array = cv2.resize(image , (160,160),3)
                    X_test.append(list(image_array))
                    y_test.append(code[folder])
            np.save('X_test',X_test)
            np.save('y_test',y_test)

            loaded_X_test = np.load('./X_test.npy')
            loaded_y_test = np.load('./y_test.npy')
            y_test = loaded_y_test
            history=tf.keras.models.load_model(self.h5file) 
            predictions=np.argmax(history.predict(loaded_X_test),axis=1)
            conf_m = confusion_matrix(y_test, np.round(predictions))
            self.acc = accuracy_score(y_test, np.round(predictions)) * 100
            plot_confusion_matrix(conf_mat = conf_m, figsize = (6, 6), cmap = matplotlib.pyplot.cm.Reds)
            plt.show()
        else:self.show_popup()
    def KayipMatrisi(self):
        plt.figure(figsize = (10, 5))
        plt.title("Model loss")
        plt.plot(self.history_resnet.history["loss"], "go-")
        plt.plot(self.history_resnet.history["val_loss"], "ro-")
        plt.legend(["loss", "val_loss"])
        plt.show()

    def DogrulukMatrisi(self):
        plt.figure(figsize = (10, 5))
        plt.title("Model accuracy")
        plt.plot(self.history_resnet.history["accuracy"], "go-")
        plt.plot(self.history_resnet.history["val_accuracy"], "ro-")
        plt.legend(["accuracy", "val_accuracy"])
        plt.show()

    def VeriCoklama(self):
        def isCheck_Klasor(path):
            if not os.path.isdir(path):
                os.mkdir(path)
        classes=os.listdir(self.test)
        datagen = ImageDataGenerator(rotation_range=180,width_shift_range=0.3,height_shift_range=0.3,
        shear_range=0.15,zoom_range=0.5,horizontal_flip=True)
        path1='C:/Users/elifd/Desktop/finalyapayzeka/Dataset/VerilerCoklanmisHali/'
        isCheck_Klasor(path1)

        for class_name in classes:
            path2=self.test+class_name+'/'
            files=os.listdir(path2)
            save_here = path1+class_name
            isCheck_Klasor(save_here)
            
            for file in files:
                image_path = path2+file

                image = cv2.imread(image_path)
                image = np.expand_dims(image, 0) 
                datagen.fit(image)
                
                if class_name=="glioma":
                    range_value=3
                elif class_name=="meningioma":
                    range_value=9
                elif class_name=="notumor":
                    range_value=12
                
                    

                for x, val in zip(datagen.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpg'),range(range_value)):
                    pass
            
                print (class_name,file, " islem bitti...")


    def image_siniflandirma(self):
        if self.file[0] != 0:
            self.X = []
            self.y = []
            for folderName in os.listdir(self.test):
                if not folderName.startswith('.'):
                    if folderName in ['glioma']:
                        label = 0
                    elif folderName in ['meningioma']:
                        label = 1
                    elif folderName in ['notumor']:
                        label = 2
                    
                        
                    
                    for image_filename in tqdm(os.listdir(self.test + folderName)):
                        img_file = cv2.imread(self.test + folderName + '/' + image_filename)
                        if img_file is not None:
                            img_file = skimage.transform.resize(img_file, (160, 160))
                            img_arr = np.asarray(img_file)
                            self.X.append(img_arr)
                            self.y.append(label)
            self.X = np.asarray(self.X)
            self.y = np.asarray(self.y)
        else: self.show_popup()

    def ResimTespiti(self):
        if self.h5file[0]!=0:
            self.image_siniflandirma()
            history=tf.keras.models.load_model(self.h5file)
            # predictions = history.predict(self.X)
            test_image = load_img(self.fname[0], target_size = (160, 160))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image / 255.0
            predictions = history.predict(test_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            if predicted_class==0:
                prediction = "glioma"
                # return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (predictions, " ===> glioma")])
            elif predicted_class==1:
                prediction = "meningioma"
                # return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (predictions, " ===> meningioma")])
            elif predicted_class==2:
                prediction = "notumor"    
                # return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (predictions, " ===> notumor")])
            else: prediction="unknown"
            confidence = predictions[0][predicted_class]
            print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
            result_str = f"Bu görüntü ===> {prediction} (Yüzde: {confidence:.3f})"
            self.ui.listWidget.addItem(result_str)

            # print(result) 
            # print(prediction)
            # plt.imshow(test_image_for_plotting)
            # if(predictions[0] > 0.5):
            #     statistic = (1.0 - predictions[0]) * 100
            #     return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (statistic, " ===> glioma")])

            # if(predictions[0] > 1.5):
            #     statistic = (2.0 - predictions[0]) * 100
            #     return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (statistic, " ===> meningioma")])
            # if(predictions[0] > 2.5):
            #     statistic = (3.0 - predictions[0]) * 100
            #     return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s" % (statistic, " ===> notumor")])
            # else:
            #     statistic = predictions[0] * 100 
            #     return self.ui.listWidget.addItems(["Bu görüntü yüzde %.3f %s"% (statistic, " ===> pituitary")])
        else:
            self.show_popup()
    def modelEgit(self):
            if self.file[0]!=0:
                # if self.ui.comboBoxModelSec.currentIndex()==1:
                #     train_datagen = ImageDataGenerator(zoom_range = 0.3,
                #                     horizontal_flip = True
                #                     )
                #     test_datagen = ImageDataGenerator()
                #     train_gen = train_datagen.flow_from_directory(
                #                 directory = self.train, 
                #                 target_size = (160, 160), 
                #                 batch_size = int(self.ui.txtBatchSize.text()), 
                #                 class_mode = 'categorical', 
                #                 shuffle=True)

                #     test_gen = train_datagen.flow_from_directory(
                #                                 directory = self.test, 
                #                                 target_size = (160, 160), 
                #                                 batch_size = int(self.ui.txtBatchSize.text()), 
                #                                 class_mode = 'categorical', 
                #                                 shuffle=True)

                #     lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=2, mode='max')
                #     cp = ModelCheckpoint(filepath='eff_model.{epoch:02d}-{val_loss:.2f}.h5',save_weights_only=True)
                #     model_EffNetB5_1101 = Sequential()
                #     model_EffNetB5_1101.add(EfficientNetB5(weights='imagenet',include_top=False, input_shape=(160,160,3)))
                #     model_EffNetB5_1101.add(layers.GlobalAveragePooling2D())
                #     model_EffNetB5_1101.add(layers.Dropout(0.5))
                #     model_EffNetB5_1101.add(layers.Dense(4,activation = 'softmax'))
                #     model_EffNetB5_1101.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
                #     history_eff = model_EffNetB5_1101.fit(
                #     train_gen, steps_per_epoch=train_gen.samples/32,epochs=int(self.ui.txtEpochSayisi.text()),        
                #     validation_data=test_gen, validation_steps=test_gen.samples // 32,callbacks=[lr, cp],verbose=1)
                #     self.ui.listWidget.addItem("Eğitilen EffNetB5 modelin sonuçları:")
                #     model1SonucLoss=history_eff[0]*100
                #     model1SonucAccuracy=history_eff[1]*100
                #     self.ui.listWidget.addItem(f'Accuracy: {np.mean(model1SonucAccuracy)}')
                #     self.ui.listWidget.addItem(f' Loss: {np.mean(model1SonucLoss)}')
                #     self.model_EffNetB5_1101.save("C:\\Users\\elifd\\Desktop\\finalyapayzeka\\Dataset\\ModelsResult\\EffNetB5FromAraYuz.h5")
                if self.ui.comboBoxModelSec.currentIndex()==1:
                    train_datagen = ImageDataGenerator(zoom_range = 0.3,horizontal_flip = True)
                    test_datagen = ImageDataGenerator()
                    train_gen = train_datagen.flow_from_directory(
                                directory = self.train, 
                                target_size = (160, 160), 
                                batch_size = int(self.ui.txtBatchSize.text()), 
                                class_mode = 'categorical', 
                                shuffle=True)

                    test_gen = test_datagen.flow_from_directory(
                                                directory = self.test, 
                                                target_size = (160, 160), 
                                                batch_size = int(self.ui.txtBatchSize.text()), 
                                                class_mode = 'categorical', 
                                                shuffle=True)

                    from keras.models import Model, Sequential
                    from keras.layers import Input, Dense, Flatten, Dropout
                    from keras.layers import Conv2D, MaxPool2D, BatchNormalization

                    model_2 = Sequential([
                    Conv2D(16, (3, 3), activation='relu', input_shape=(160, 160, 3)),
                    MaxPool2D((2, 2)),
                        
                    Conv2D(32, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPool2D(pool_size=(2, 2)),

                    Conv2D(64, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPool2D(pool_size=(2, 2)),

                    Conv2D(128, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPool2D(pool_size=(2, 2)),
                    Dropout(rate=0.2),
                        
                    Conv2D(256, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPool2D(pool_size=(2, 2)),
                    Dropout(rate=0.2),

                    Flatten(),
                    Dense(units=1024, activation='relu'),
                    Dropout(rate=0.3),

                    Dense(units=3, activation='softmax') ])

                    model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
                    cp = ModelCheckpoint(filepath='basit_model.{epoch:02d}-{val_loss:.2f}.h5',save_weights_only=True)

                    self.history_resnet = model_2.fit(train_gen, 
                            steps_per_epoch=train_gen.samples/512, 
                            epochs=int(self.ui.txtEpochSayisi.text()),        
                            validation_data=test_gen, 
                            validation_steps=test_gen.samples // 512,
                            callbacks = [cp])
                    model_2.save("C:\\Users\\elifd\\Desktop\\finalyapayzeka\\Dataset\\ModelsResult\\ResNet.h5")
                    self.ui.listWidget.addItem("Eğitilen ResNet modelin sonuçları:")
                    modelLossResult=self.history_resnet.history['loss']
                    modelAccResult=self.history_resnet.history['accuracy']
                    modelValAccResult=self.history_resnet.history['val_accuracy']
                    modelValLossResult=self.history_resnet.history['val_loss']
                    self.ui.listWidget.addItem(f'Accuracy: {np.mean(modelAccResult)}')
                    self.ui.listWidget.addItem(f'Loss: {np.mean(modelLossResult)}')
                    self.ui.listWidget.addItem(f'Val Accuracy: {np.mean(modelValAccResult)}')
                    self.ui.listWidget.addItem(f'Val Loss: {np.mean(modelValLossResult)}')
            

                # if self.ui.comboBoxModelSec.currentIndex()==4:
                #     ourModel = Sequential([

                #     Conv2D(16, (2, 2), activation='relu', input_shape=(160, 160, 3)),
                #     MaxPool2D((2, 2)),

                #     Conv2D(32, (3, 3), activation='relu'),
                #     BatchNormalization(),
                #     MaxPool2D(pool_size=(2, 2)),

                #     Conv2D(64, (3, 3), activation='relu'),
                #     BatchNormalization(),
                #     MaxPool2D(pool_size=(2, 2)),

                #     Conv2D(128, (2, 2), activation='relu'),
                #     BatchNormalization(),
                #     MaxPool2D(pool_size=(3, 3)),
                #     Dropout(rate=0.4),

                #     Flatten(),
                #     Dense(units=1024, activation='relu'),
                #     Dropout(rate=0.3),

                #     Dense(units=4, activation='softmax') ])

                #     ourModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

                #     cp = ModelCheckpoint(filepath='basit_model.{epoch:02d}-{val_loss:.2f}.h5',save_weights_only=True)
                #     history_ourModel = ourModel.fit(
                #                train_gen, 
                #                steps_per_epoch=train_gen.samples/24, 
                #                epochs=5,        
                #                validation_data=test_gen, 
                #                validation_steps=test_gen.samples // 24,
                #                callbacks = [cp])
                #     ourModel.save("C:\\Users\\Aiperi\\Desktop\\BrainTumorProject\\Dataset\\ModelsResult\\ResNet.h5")")
                #     readyOurModel=load_model("/content/drive/MyDrive/Dataset/ourModel.h5")    
                else:self.show_popup()
    def KlasorAc(self):
        self.file = str(QFileDialog.getExistingDirectory(self, "Dosya seç"))
        self.ui.listWidget.addItem("Seçilen dosya adı :"+self.file)
        self.train=self.file+"/train/"
        self.test=self.file+"/test/"

    def ResimSec(self):
        self.fname=QFileDialog.getOpenFileName(self,"Resim seç","", "RESIMLER (*.png;*.jpg;*.jpeg)")
        self.pixmap=QPixmap(self.fname[0])
        scaled = self.pixmap.scaled(self.ui.imageBox.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.imageBox.setPixmap(scaled)
        sp = self.ui.imageBox.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.ui.imageBox.setSizePolicy(sp)
        self.layout().setAlignment(self.ui.imageBox, QtCore.Qt.AlignCenter)

    def h5Sec(self):
        self.h5file,_=QFileDialog.getOpenFileName(self,"h5 seç","", "h5 (*.h5)")
        self.ui.listWidget.addItem("Seçilen h5 dosyası:"+self.h5file)    
    
    def Siniflandir(self):
        if self.file[0] != 0:
            if self.ui.comboBoxSinif.currentIndex()!=0:
                X_train = []
                y_train = []
                code = {'glioma':0 ,'meningioma':1,'notumor':2}
                for folder in  os.listdir(self.train) : 
                    files = glob.glob(pathname= str(self.train + folder + '/*.jpg'))
                    for file in files: 
                        image = cv2.imread(file)
                        #resize images to 64 x 64 pixels
                        image_array = cv2.resize(image , (160,160))
                        X_train.append(list(image_array))
                        y_train.append(code[folder])
                np.save('X_train',X_train)
                np.save('y_train',y_train)

                X_test = []
                y_test = []
                for folder in  os.listdir(self.test) : 
                    files = glob.glob(pathname= str(self.test + folder + '/*.jpg'))
                    for file in files: 
                        image = cv2.imread(file)
                        image_array = cv2.resize(image , (160,160))
                        X_test.append(list(image_array))
                        y_test.append(code[folder])
                np.save('X_test',X_test)
                np.save('y_test',y_test)


                loaded_X_train = np.load('./X_train.npy')
                loaded_X_test = np.load('./X_test.npy')
                loaded_y_train = np.load('./y_train.npy')
                loaded_y_test = np.load('./y_test.npy')

                X_train = loaded_X_train.reshape([-1, np.product((160,160,3))])
                X_test = loaded_X_test.reshape([-1, np.product((160,160,3))])
                y_train = loaded_y_train
                y_test = loaded_y_test
                X_train, y_train = shf(X_train, y_train, random_state=15)
                X_test, y_test = shf(X_test, y_test, random_state=15)

                if self.ui.comboBoxSinif.currentIndex()==1:
                    knn = KNeighborsClassifier(n_neighbors=10)
                    knn.fit(X_train, y_train)
                    knn_predcited = knn.predict(X_test)
                    self.ui.listWidget.addItem("KNN accuracy score is: " + str(knn.score(X_test, y_test)))
                    #self.plot_cm(knn_predcited, y_test, "KNN Confusion Matrix")
                    labels = ['glioma', 'meningioma', 'notumor']
                    cm = confusion_matrix(y_test,knn_predcited)
                    cm = pd.DataFrame(cm , index = ['0','1','2'] , columns = ['0','1','2'])
                    plt.figure(figsize = (7,7))
                    plt.title("KNN Confusion Matrix")
                    sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                    plt.show()
                
                if self.ui.comboBoxSinif.currentIndex()==2:
                    log_reg  = LogisticRegression(solver='lbfgs', max_iter=100)
                    log_reg.fit(X_train, y_train)
                    log_reg_predcited = log_reg.predict(X_test)
                    self.ui.listWidget.addItem('Logistic Regression accuracy score is: ' + str(log_reg.score(X_test, y_test)))
                    #self.plot_cm(log_reg_predcited, y_test, 'Logistic Regression Confusion Matrix') 
                    labels = ['glioma', 'meningioma', 'notumor']
                    cm = confusion_matrix(y_test,log_reg_predcited)
                    cm = pd.DataFrame(cm , index = ['0','1','2'] , columns = ['0','1','2'])
                    plt.figure(figsize = (7,7))
                    plt.title("Logistic Regression Confusion Matrix")
                    sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                    plt.show()

                if self.ui.comboBoxSinif.currentIndex()==3:
                    svm = SVC(max_iter=100)
                    svm.fit(X_train, y_train)
                    svm_predcited = svm.predict(X_test)
                    self.ui.listWidget.addItem('Support Vector Machine Classifier accuracy score is: ' + str(svm.score(X_test, y_test)))
                    #self.plot_cm(svm_predcited, y_test, 'Support Vector Machine Confusion Matrix')
                    labels = ['glioma', 'meningioma', 'notumor']
                    cm = confusion_matrix(y_test,svm_predcited)
                    cm = pd.DataFrame(cm , index = ['0','1','2'] , columns = ['0','1','2'])
                    plt.figure(figsize = (7,7))
                    plt.title("Support Vector Machine Confusion Matrix")
                    sns.heatmap(cm, linecolor = 'black' , linewidth = 1 , annot = True, fmt='', xticklabels = labels, yticklabels = labels)
                    plt.show()    
            else:
                msg=QMessageBox()
                msg.setWindowTitle("Uyarı!")
                msg.setText("Siniflandirici seçiniz!")
                x=msg.exec_()
        else:
            self.show_popup()

    def SelectKFold(self):
        if self.ui.comboBoxKFold.currentIndex()==1:
            return 5
        elif self.ui.comboBoxKFold.currentIndex()==2:
            return 10
        elif self.ui.comboBoxKFold.currentIndex()==3:
            return 15    

    def show_popup(self):
        if self.file[0]==0:
            msg=QMessageBox()
            msg.setWindowTitle("Uyarı!")
            msg.setText("Dosya seç !")
            x=msg.exec_()
        elif self.h5file[0]==0:
            msg=QMessageBox()
            msg.setWindowTitle("Uyarı!")
            msg.setText("H5 file seç!")
            x=msg.exec_()
        elif self.fname[0]==0:
            msg=QMessageBox()
            msg.setWindowTitle("Uyarı!")
            msg.setText("Resim  seç!")
            x=msg.exec_()
def app():
    app=QtWidgets.QApplication(sys.argv)
    win=window()
    win.show()
    sys.exit(app.exec_())

app()

