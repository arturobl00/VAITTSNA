#nota para hacer este proyecto tome un dataset de kaggle
#https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset?select=Face+Mask+Dataset

import cv2
import os
import numpy as np

#Ruta donde estan las imagenes
dataPath = "C:/Users/BUSTAMANTE/Google Drive/TALLER DE VISION ART/MASCARILLAS/dataset"
dir_list = os.listdir(dataPath)
print("Lista de archivos:", dir_list)

#variables que usaremos
labels = []
faceData = []
label = 0

#Creamos un ciclo para recorrer los archivos y obtener la ruta y dentro un ciclo anidado para sumar la ruta al nombre del archivo
for name_dir in dir_list:
    dir_path = dataPath + "/" + name_dir
    #Ciclo anidado con la dir_path solo falta el nombre del archivo y lo mostramos para ver si funciona
    for file_name in os.listdir(dir_path):
        image_path = dir_path + "/" + file_name
        print(image_path)
        imageOriginal = cv2.imread(image_path)
        hsv_green = cv2.cvtColor(imageOriginal,cv2.COLOR_BGR2HSV)
        image = cv2.imread(image_path,0)
        cv2.imshow("Imagen Procesada", image)
        cv2.imshow("Imagen Original", imageOriginal)
        cv2.imshow("Imagen HSV", hsv_green)
        cv2.waitKey(50)

        #ahora vamos almacenar en faceData todas las imagenes y en label todas las etiquetas
        faceData.append(image)
        labels.append(label)
    label += 1

#para comprobar cuantos datos tenemos en la label 0 y en la label 1
print("Label 0: ", np.count_nonzero(np.array(labels) == 0))
print("Label 1: ", np.count_nonzero(np.array(labels) == 1))

#ahora vamos a hacer un reconocimiento con LBPH esto va a generar un modelo que es un archivo xml
face_mask = cv2.face.LBPHFaceRecognizer_create()

#Entrenamiento
print("Entrenando Modelo...",face_mask)
face_mask.train(faceData, np.array(labels))

#Almacenar modelo
face_mask.write("face_mask_model.xml")
print("Modelo almacenado archivo face_mask_model.xml")


