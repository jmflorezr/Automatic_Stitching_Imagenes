import numpy as np

from calcular import *
import matplotlib.pyplot as plt

#esta funcion toma 2 imagenes y las suma
def sumarImg(image1,image2):
    image1 = (image1/255).astype(np.float)
    image2 = (image2 / 255).astype(np.float)

    return (image1+image2)/2


if __name__ == '__main__':
    path = sys.argv[1]

    # en esta lista estan todas las imagenes disponibles, y la Referencia elegida
    listaImages, pos = getImages(path)
    #imgzeros = np.zeros((listaImages[0].shape[0], listaImages[0].shape[1], 3), np.uint8)

    # getHomografia(), recibe lista con imagenes y el metodo(0=shift,1=orb)
    listaHomografia = getHomografia(listaImages, 0)

    listaT = trasforImages(listaHomografia, listaImages, pos)
    suma1 =sumarImg(listaT[0], listaT[1]) #suma las 2 imagenes de la trasformacion

    cv2.imshow("suma ", suma1)
    cv2.waitKey(0)

