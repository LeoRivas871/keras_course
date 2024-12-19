from keras import models
from keras import layers
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
network = models.Sequential() #Se crea una instancia de la clase Sequential. Esto significa que se esta creando una red
#neuronal que tendra capas apiladas en secuencia.

network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) #Se agrega la primera capa a la red neuronal usando network.add().
#Esta capa es una capa Dense (densa o completamente conectada) con las siguientes caracteristicas:
#512 unidades: La capa tiene 512 neuronas.
#Función de activación 'relu': Introduce no linealidad en la red, lo que le permite aprender patrones más complejos.
#input_shape=(28*28): Define la forma de la entrada que recibira la capa. En este caso se espera una entrada unidimensional de
#784 elementos (28*28), que correspondería a una imagen de 28*28 pixeles aplanada en un vector.

network.add(layers.Dense(10,activation='softmax')) #Se agrega la segunda y última capa a la red. Esta capa también es una Dense con:
#10 unidades: Representa las 10 posibles clases de salida (dígitos del 0 al 9).
#Función de activación 'softmax': La función softmax convierte las salidas de la capa en probabilidades, donde cada valor representa
#la probabilidad de que la entrada pertenezca a una de las 10 clases.
'''En resumen, este código define una red neuronal sencilla para clasificar imagenes de digitos escritos a mano.
La red consta de dos capas densas, una con 512 neuronas y activacion ReLu, y otra con 10 neuronas y activación softmax. 
La Red recibe una imagen de 28*28 pixeles como entrada, la procesa a través de las capas y produce una salida de 10 probabilidades,
una para ca dígito del 0 al 9. La clase con la probabilidad más alta se considera la predicción de la red.
'''
#La red de nuestro ejemplo consta de una secuencia de dos capas Dense, que son capas neuronales densamente conectadas o
#completamente conectadas. La segunda (y ultima capa) es una capa "softmax" de 10 neuronas, lo que significa que devolverá una
#matriz de 10 resultados de probabilidad (sumando 1). Cada resultado será la probabilidad de que la imagen del dígito actual 
#pertenezca a una de nuestras diez clases de números.

'''Para preparar la red para el entrenamiento, necesitamos coger otras tres cosas en el paso de compilación:
---Una funcion de perdida: Sirve para determinar como podra la red medir su rendimiento con los datos de entrenamiento, y por consiguiente, como podra tomar
la desición correcta.
---Un optimizador: El mecanismo por el cual se actualizará la red basandose en los datos que ve y su función de perdida.
---Metricas para monitorizar el entrenamiento y las pruebas: Aquí solo nos preocuparemos por la exactitud (accuracy), la fracción de las
imagenes que fueron clasificadas correctamente. '''

#Paso de compilación
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

'''Antes de entrenar, procesaremos los datos dándoles la forma que la red espera y escalándolos de modo que todos los valores queden en el intervalo [0,1]. Antes,
nuestras imágenes de entrenamiento, por ejemplo, estaban almacenadas en una matriz de forma (60000,28,28) de tipo uint8 con valores en el intervalo [0,255].
La vamos a transformar en una matriz de forma (60000,28*28) de tipo float32 con valores en el intervalo [0,1].'''
#Preparacion de los datos de imagen.
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

#Preparación de las etiquetas.
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Ya estamos listos para entrenar a la red, lo cual se hace en Keras a través de una llamada al metodo fit de la red,
#que ajusta el modelo a sus datos de entrenamiento:
history = network.fit(train_images,train_labels,epochs=5,batch_size=128)
test_loss,test_acc = network.evaluate(test_images,test_labels)
print(f'Test accuracy: {test_acc}')
print(f'Resultado: {history.history}')
