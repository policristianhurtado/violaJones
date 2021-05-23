# Importaciones
import cv2 as cv        # OpenCV
import os               # Libreria para interacturar con carpetas del sistema

# Definimos la ruta en donde se encuentran los datos de los rostros
dataPath = "C:/Users/Angry/Documents/python/prueba\openCV/reconocimiento_facial/Data"
# Optenemos los nombres de las personas por medio del nombre de las carpetas
imagePaths = os.listdir(dataPath)
print(imagePaths)

# Para realizar el reconocimiento de los rostros conocemos tres forams de realizarlas
# por medio de Eigen faces, Fisher Faces y LBPHF Faces, en este caso nosotros vamos a
# usar Eigen Faces, pero dejaremos el codigo de como seria con las otras opciones.

# Eigen
# Definimos una variable para realizar el reconocimiento de la persona
face_recognizer = cv.face.EigenFaceRecognizer_create()

# Fisher
'''
face_recognizer = cv.face.FisherFaceRecognizer_create()
'''

# lBPHF
'''
face_recognizer = cv.face.LBPHFaceRecognizer_create()
'''

# Leemos en modelo de entrenamiento .xml ya guardado
# Eigen
face_recognizer.read("modelEigenFace.xml")

# lBPHF
'''
face_recognizer.read("modelFisherFace.xml")
'''

# Fisher
'''
face_recognizer.read("modelLBPHFFace.xml")
'''

# Abrimos el recurso o abrimos la camara donde se analizara si la persona/s esta/n entrenadas
# para un video seria de la siguiente forma
'''
cap = cv.VideoCapture("C:/Path")
'''
# Y para activar la camara seria
'''
cap = cv.VideoCapture(0)
'''

# Se llama al archivo preentrenado
cascadePath = "haarcascade_frontalface_default.xml"

# Se asigna a una variable con visiones de computador
face_classif = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

# Definimos un while para ejecutarse si el video tiene alguna informacion, sino pues
# Se cierra
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    # Tranformamos el video(captura) en escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Copiamos la informacion de la captura en otra variable
    aux_frame = gray.copy()

    # Se ajustan los parametros para la detección
    faces = face_classif.detectMultiScale(
        gray,
        scaleFactor=1.09,
        minNeighbors=5
    )
    # Realizamos un ciclo para determinar los rostros que existen en el video(captura)
    for (x, y, w, h) in faces:
        # Agregamos la ubicacion del rostro en [x, y]
        rostro = aux_frame[y:y+h, x:x+w]
        # Definimos el tamaño que tomara el rostro
        rostro = cv.resize(rostro, (150, 150), interpolation=cv.INTER_CUBIC)
        # analizamos si la persona corresponde o no los datos almacenados
        result = face_recognizer.predict(rostro)
        # Agregamos el resultado en la visualizacion del video(captura)
        cv.putText(frame, '{}'.format(result), (x, y-5),
                   1, 1.3, (255, 255, 0), 1, cv.LINE_AA)

        # Con respecto al valor del resultado definimos si existe alguna concordancia
        # con los datos almacenados
        # Eigen
        # Si el resultado es menor a 5700 es la persona y si no la persona es desconicida
        if result[1] < 5700:
            # Agregamos el nombre de la persona al video(captura)
            cv.putText(frame, '{}'.format(
                imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            # Agregamos el rectangulo donde se encuentra el rostro de la persona
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Agregamos el texto desconocido al video(captura)
            cv.putText(frame, "Desconocido", (x, y-20),
                       2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            # Agregamos el rectangulo donde se encuentra el rostro de la persona
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Fisher
        '''
        if result[1] < 500:
            cv.putText(frame, '{}'.format(
                imagePaths[result[0]]),(x,y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        else:
            cv.putText(frame, "Desconocido",(x,y-20), 2,
                       0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        '''
        # lBPHF
        '''
        if result[1] < 70:
            cv.putText(frame, '{}'.format(
                imagePaths[result[0]]),(x,y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        else:
            cv.putText(frame, "Desconocido",(x,y-20), 2,
                       0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        '''
    # Visualizamos el video(captura)
    cv.imshow("frame", frame)
    # Espesificamos el cierre del video(captura)
    if cv.waitKey(1) & 0xFF == ord('s'):
        break
    #k = cv.waitKey(1)
    # if k == 25:
    #    break

# Cerramos y "destruimos" las ventanas creadas
cap.release()
cv.destroyAllWindows()
