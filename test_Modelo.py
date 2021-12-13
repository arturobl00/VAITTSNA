import cv2
import os
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

#Creando una constante
LABELS = ["Con_Mascarilla", "Sin_Mascarilla"]

#Leer el modelo creado pero damos de alta la variable
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

#Prendemos nuestra camara
cap = cv2.VideoCapture(0)

#Inicamos la deteccion de rostros
while True:
    ret, frame = cap.read()
    negro = np.zeros(frame.shape, np.uint8)
    mix = np.zeros(frame.shape, np.uint8)
    if ret == False:
        break
    
    with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

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

                mp_drawing.draw_detection(frame_rgb, detection)
                mp_drawing.draw_detection(mix, detection)
                
                #Ahora en una nueva ventana crearemos la captura de nuestro rostro detectado a 72 x 72
                face_image = frame[ymin : ymin + h, xmin : xmin + w]
                hsv_green = cv2.cvtColor(face_image,cv2.COLOR_BGR2HSV)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                
                face_image1 = face_image
                face_image1 = cv2.resize(face_image, (300, 300), interpolation= cv2.INTER_CUBIC)
                hsv_green = cv2.resize(hsv_green, (300, 300), interpolation= cv2.INTER_CUBIC)
                #negro = cv2.resize(negro, (300, 300), interpolation= cv2.INTER_CUBIC)
                face_image = cv2.resize(face_image, (72, 72), interpolation= cv2.INTER_CUBIC)
                
                #Ahora a poner en marcha nuestro entrenamiento
                result = face_mask.predict(face_image)
                cv2.putText(frame, "$#Search-{}".format(result), (xmin - 100, ymin + 200), 1, 1.3, (255,255,255), 1, cv2.LINE_AA )
                cv2.putText(face_image1, "{}".format(result), (xmin - 150, ymin), 1, 1, (255,255,255), 1, cv2.LINE_AA )
                cv2.putText(hsv_green, "TRACK{}".format(result), (xmin - 150, ymin), 1, 1, (0,0,0), 1, cv2.LINE_AA )
                if result[1] < 150:
                    color = (0, 255, 0) if LABELS[result[0]] == "Con_Mascarilla" else (0, 0, 255)
                    cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin -25), 2, 1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces = 2,
        min_detection_confidence=0.5) as face_mesh:
    
        result1 = face_mesh.process(frame_rgb)
        
        if result1.multi_face_landmarks is not None:
        #Si tenemos informacion entonces con un form recorremos los datos y dibujamos
            for face_landmark in result1.multi_face_landmarks:
                #Dibujar los 21 puntos de deteccion
                mp_drawing.draw_landmarks(
                negro, face_landmark, mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1))

                mp_drawing.draw_landmarks(
                mix, face_landmark, mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))
                
        cv2.imshow("Puntos Rostro",frame_rgb)
        cv2.imshow("Detector de rostro", frame)
        cv2.imshow("Imagen Sensada", hsv_green)
        cv2.imshow("Face_image", face_image1)
        cv2.imshow("Malla de Rostro", negro)
        cv2.imshow("Mix Sensor", mix)

    k = cv2.waitKey(1)
    if k == 13:
        break

cap.realise()
cv2.destroyAllWindows()