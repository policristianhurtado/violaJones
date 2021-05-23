# Importación de Librerias
import cv2 as cv    # OpenCV
import os           # Libreria para interacturar con carpetas del sistema
import imutils      # Libreria para realizar tareas básicas de procesamiento de imágenes

# Especificamos el nombre de la persona de la cual vamos a entrenar los rostros
personName = "cristian"
# Especificamos la ruta en la cual se va a almacenar dichos datos
dataPath = "C:/Users/Angry/Documents/python/prueba/openCV/reconocimiento_facial/Data"
# Unimos el nombre de la persona con la ruta de almacenamiento
personPath = dataPath + "/" + personName

# Validamos si la carpeta de la persona ya existe, si no es asi la creamos
if not os.path.exists(personPath):
    print("Carpeta Creada: ", personPath)
    os.makedirs(personPath)

#Realizamos la captura de los rostros por medio de la camara
'''
cap = cv.VideoCapture(0)
'''

# O lo podemos hacer por medio de un video, por medio del siguiente codigo.
'''
cap = cv.VideoCapture("C:/Path del video para entrenar")
'''
#Se llama al archivo preentrenado
cascadePath = "haarcascade_frontalface_default.xml"

# Se asigna a una variable con visiones de computador
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

#Definimos una variable para realizar la contabilizacion de los rostros almacenados
count = 0

# Definimos un While para realizar la deteccion y alamcenamiento de los rostros
while True:
    # Almacenamos los resultados del video en dos variables
    ret, frame = cap.read()
    if ret == False : break
    # definimos el tamaño de visualizacion del video
    frame = imutils.resize(frame, width=640)
    # Transformamos el video en escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Copiamos la informacion del video(captura) en otra variable
    auxFrame = frame.copy()
    # Se ajustan los parametros para la detección
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    # Realizamos un ciclo para determinar los rostros que existen en el video(captura)
    for (x,y,w,h) in faces:
        # Dibujamos el rectangulo sobre el rostro detectado
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        # Definimos el tamaño para todos los rostros, es decir que se almacenen en
        # imagenes de 150px X 150px, esto para que todas las imagenes me queden del 
        # mismo tamaño
        rostro = cv.resize(rostro,(150,150), interpolation=cv.INTER_CUBIC)
        # Se especifica el nombre que recibirá la imagen al momento de almacenarse
        cv.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        # El contabilizador aumenta en 1
        count = count + 1
    
    # Visualizamos el video(captura)
    cv.imshow('frame', frame)
    
    # Definimos el cierre del video cuando el contado llegue a 300
    # es decir que se almacenaran 300 imagenes del rostro de la persona
    # a entrenar
    k = cv.waitKey(1)
    if k == 27 or count >= 300:
        break

# Cerramos todas las ventanas abiertas
cap.release()
cv.destroyAllWindows()
