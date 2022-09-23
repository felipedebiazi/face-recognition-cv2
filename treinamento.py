import os
import imutils
import numpy as np
import cv2

## PREPARANDO O DATASET

dataPath = 'D:/Imagens/Dataset_Fotos\Recognizer' # Troque aqui o endereço seu dataset.
peopleList = os.listdir(dataPath)
print('Lista de pessoas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Lendo Imagens')

    for fileName in os.listdir(personPath):
        #print('Rostos: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        image = cv2.resize(image,(150,150),interpolation=cv2.INTER_CUBIC)
        ##cv2.imshow('image',image)
        ##cv2.waitKey(10)    
        
    label = label + 1
 
## print('labels= ',labels)
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))


## METODOS DE TREINAMENTO
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

## TREINANDO O RECOGNIZER
print("Treinando...")
face_recognizer.train(facesData, np.array(labels))

## ARMAZENANDO MODELO OBTIDO 
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo Salvo...")