import cv2
import os
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

#Creando una constante
LABELS = ["Con_Mascarilla", "Sin_Mascarilla"]

#Leer el modelo creado pero damos de alta la variable
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

#Prendemos nuestra camara
cap = cv2.VideoCapture(0)

#Inicamos la deteccion de rostros
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        if ret == False: break

        frame = cv2.flip(frame,1)
        
        #obtenemos el tamaño y convertimos a RGB
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(frame_rgb)

        #Crear un rectangulo para la deteccion del rostro
        if result.detections is not None:
            for detection in result.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                
                #Solo lo ocupamos para ver que nos cree el rectangulo detectando rostro luego lo usaremos abajo
                #cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0,255,0), 5)

                #Ahora en una nueva ventana crearemos la captura de nuestro rostro detectado a 72 x 72
                face_image = frame[ymin : ymin + h, xmin : xmin + w]
                hsv_green = cv2.cvtColor(face_image,cv2.COLOR_BGR2HSV)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                

                face_image1 = cv2.resize(face_image, (300, 300), interpolation= cv2.INTER_CUBIC)
                hsv_green = cv2.resize(hsv_green, (300, 300), interpolation= cv2.INTER_CUBIC)

                face_image = cv2.resize(face_image, (72, 72), interpolation= cv2.INTER_CUBIC)
                
                #Solo lo ocupamos para ver si nos redimienciona bien
                #cv2.imshow("Face_image", face_image)

                #Ahora a poner en marcha nuestro entrenamiento
                result = face_mask.predict(face_image)
                cv2.putText(frame, "$#Search-{}".format(result), (xmin - 100, ymin + 200), 1, 1.3, (255,255,255), 1, cv2.LINE_AA )
                cv2.putText(face_image1, "{}".format(result), (xmin - 150, ymin), 1, 1, (255,255,255), 1, cv2.LINE_AA )
                cv2.putText(hsv_green, "TRACK{}".format(result), (xmin - 150, ymin), 1, 1, (0,0,0), 1, cv2.LINE_AA )
                if result[1] < 150:
                    color = (0, 255, 0) if LABELS[result[0]] == "Con_Mascarilla" else (0, 0, 255)
                    cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin -25), 2, 1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                
        cv2.imshow("Detector de rostro", frame)
        cv2.imshow("Imagen Sensada", hsv_green)
        cv2.imshow("Face_image", face_image1)

        k = cv2.waitKey(1)
        if k == 13:
            break

cap.realise()
cv2.destroyAllWindows()