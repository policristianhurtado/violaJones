# Importaciones
import cv2 as cv        # OpenCV
import os               # Libreria para interacturar con carpetas del sistema
import numpy as np      # Libreria para realizar tareas básicas de procesamiento de imágenes

# Definimos una variable con la ruta de los datos guardados de los rostros de la persona a analizar
dataPath = "C:/Users/Angry/Documents/python/prueba\openCV/reconocimiento_facial/Data"
# Dado que se almacena una carpeta por persona, aqui se saca el nombre de cada carpeta
peopleList = os.listdir(dataPath)

# Se definen dos variables para guardar los datos de los rostros y a que person pertenece dichos rostros
labels = []
facesData = []

# Se crea una variable para guardar la persona a quien pertenece el rostro, 
# es decir, la primera persona tendra el valor 0, la segunda el valor de 1 y asi 
# hasta llegar a la cantidad de personas de las cuales se tienen los rostros almacenados
label = 0

# Un ciclo para leer las imagenes de los rostros almacenados
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print("Leyendo las imagenes")
    
    # En este for se agregara a las variables labels y facesData
    # la informacion de aquien pertenece y sus rostros, ademas de 
    # visualizar los rostros que se van guardando en dichas variables
    for fileName in os.listdir(personPath):
        print("Rostros: ", nameDir + "/" + fileName)
        labels.append(label)
        facesData.append(cv.imread(personPath+'/'+fileName, 0))
        image = cv.imread(personPath+'/'+fileName,0)
        cv.imshow("image", image)
        cv.waitKey(10)
    # Luego de que termien con la primera persona aumentara en uno el contador label
    label=label+1

# En esta seccion se definira el metodo de entrenamiento para los rostros, 
# existen 3 formas que permite la libreria OpenCV, por medio de EigenFaces,
# Fisher Faces y LBPHF Faces, nosotros vamos a utilizar Eigen Faces para realizar esta
# Prueba, pero se dejara como seria si se quiere hacer por medio de las otras dos opciones

# Eigen Face
print("Entrenendo a las personas.....")
# Creamos una variable para realizar el entrenamiento de los rostros
face_recognizer = cv.face.EigenFaceRecognizer_create()
# Entrenando al reconocedor de rostros
face_recognizer.train(facesData, np.array(labels))
# Almacenando el modelo, el nombre puede ser el que se requiera,
# En este caso escogimos el nombre de modelEigenFace y la extension debe
# ser .xml
face_recognizer.write("modelEigenFace.xml")
print("Modelo almacenado....")

#Fisher Face
'''
print("Entrenendo a las personas.....")
face_recognizer = cv.face.FisherFaceRecognizer_create()
# Entrenando al reconocedor de rostros
face_recognizer.train(facesData, np.array(labels))
# Almacenando el modelo
face_recognizer.write("modelFisherFace.xml")
print("Modelo almacenado....")
'''

#LBPHF Face
'''
print("Entrenendo a las personas.....")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Entrenando al reconocedor de rostros
face_recognizer.train(facesData, np.array(labels))
# Almacenando el modelo
face_recognizer.write("modelLBPHFFace.xml")
print("Modelo almacenado....")
'''