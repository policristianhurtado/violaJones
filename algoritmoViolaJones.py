# Importación de Librerias.
import cv2          # OpenCV
import sys          # Libreria para proveer variables y funciones
import time         # Para tomar el tiempo de ejecucion de dicho algoritmo
import imutils      # Para definir el tamaño de la pantalla de la imagen a visualizar

# definir una funcion start para que empiece a contabilizar el tiempo de ejecución
def  start():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    
# definir funcion end para que nos muestre el resultado del tiempo de ejecución
def end():    
    print(
        "El tiempo de ejecución es: " +
        str(time.time() - startTime_for_tictoc) +
        " segundos"
    )

# Llamamos a la funcion start()    
start()

#Se llama al archivo preentrenado
cascPath = "haarcascade_frontalface_default.xml"

# Se asigna a una variable con visiones de computador
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+cascPath)

#llamamos el recurso a analizar
image = cv2.imread('C:/Users/Angry/Documents/python/prueba/openCV/reconocimiento_facial/resources/people.png')

# Cambiamos a escala de grises la imagen anteriormente definida
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Se ajustan los parametros para la detección
# scaleFactor = Escala de la distancia del rostro de la persona, entre mas pequeño el valor mas 'distancia' tendra
# minNeighbors = Cantidad de recuentos de los rostros encontrados,
# minSize = Tamaño minimo del rostro
faces = faceCascade.detectMultiScale(
    img_gray,
    scaleFactor = 1.1,
    minNeighbors=5,
    minSize=(20,20)
)

# Imprimimos el numero de rostros que se han encontrado
print("Se encontraron {0} rostros!!".format(len(faces)))

# Dibujamos un rectangulo en la ubicación del rostro
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
    
end()

#Definimos el tamaño de la visualización de la imagen
image = imutils.resize(image, width=1020)
cv2.imshow("Viola Jones", image)
cv2.waitKey(0)