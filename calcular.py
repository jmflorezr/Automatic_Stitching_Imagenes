import cv2
from enum import Enum
import numpy as np
import sys
import os
import matplotlib.pyplot as plt




def getImages(path):
    contenido = os.listdir(path)
    dic_img = {}
    for i in range(len(contenido)):
        dic_img[i] = contenido[i]

    print("Imagnes disponibles: ------------>")

    for i in range(len(dic_img)):
        print(i, dic_img[i])

    sel = int(input("¿cual imagen desea usar como referencia? "))  # eleccion de imagen
    if not (sel >= 0 and sel < len(contenido)):
        print("Debe elegir una opcion valida: ")
        return False
    listImages=[]
    for i in range(0, len(dic_img)):
        image = cv2.imread(path + '\\' + dic_img[i])  # lee la imagen1
        listImages.append(image)

    return listImages, sel

"""Este metodo recibe las imagenes y el metodo(shift / orb), encuentra los puntos de interes
calcula la homografia y devuelve lista con las homografias para N imagenes de entrada.
las Homografias que retorna son N-1"""
def getHomografia(listaImages, Metodo):

    # listImagesWarped = []
    listaHomografia = []
    for i in range(0, len(listaImages)-1):
        image_gray_1 = cv2.cvtColor(listaImages[i], cv2.COLOR_BGR2GRAY)
        image_gray_2 = cv2.cvtColor(listaImages[i+1], cv2.COLOR_BGR2GRAY)

        # sift/orb interest points and descriptors

        if Metodo == 0:
            sift = cv2.SIFT_create(nfeatures=100)  # shift invariant feature transform
            keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
        else:
            orb = cv2.ORB_create(nfeatures=100)  # oriented FAST and Rotated BRIEF
            keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
            keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)

        image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
        image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)


        # Interest points matching
        bf = cv2.BFMatcher(cv2.NORM_L2)#agregue crossCheck
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
        image_matching = cv2.drawMatchesKnn(listaImages[i], keypoints_1, listaImages[i+1], keypoints_2, matches, None)
        cv2.imshow("image matching",image_matching)
        cv2.waitKey(0)
        # Retrieve matched points
        points_1 = []
        points_2 = []
        for idx, match in enumerate(matches):
            idx2 = match[0].trainIdx
            points_1.append(keypoints_1[idx].pt)#esto estaba invertido
            points_2.append(keypoints_2[idx2].pt)#esto estaba invertido

        # Compute homography and warp image_1
        H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)
        listaHomografia.append(H)
        # if i==0:
        #     image_warped = cv2.warpPerspective(listaImages[i], H, (listaImages[i].shape[1]+listaImages[i+1].shape[1], listaImages[i].shape[0]))
        #     #image_warped[0:listaImages[i].shape[0], 0:listaImages[i].shape[1]] = listaImages[i]
        # elif i==1:
        #     image_warped = cv2.warpPerspective(listaImages[i+1], np.linalg.inv(H), (listaImages[i].shape[1]+listaImages[i+1].shape[1], listaImages[i].shape[0]))
        # #image_warped = cv2.warpPerspective(listaImages[i+1], H,(listaImages[i].shape[1] + listaImages[i + 1].shape[1], listaImages[i].shape[0]))
        #     image_warped[0:listaImages[i].shape[0], 0:listaImages[i].shape[1]] = listaImages[i]
        # listImagesWarped.append(image_warped)


    return listaHomografia


def trasforImages(H, listaImages, pos):
    listaImagesTrasformadas =[]
    alto,ancho =listaImages[0].shape[:2]
    print(alto,"--------",ancho)

    if pos==0: # #Stitchin de derecha --->izq
        image_warped = cv2.warpPerspective(listaImages[1], np.linalg.inv(H[0]), (ancho*2, alto))#trasforma 1 a 0
        image_warped[0:listaImages[0].shape[0], 0:listaImages[0].shape[1]] = listaImages[0]# union de ref con imagen trasformada 1 a 0

        listaImagesTrasformadas.append(image_warped) #agrega a lista
        image_warped = cv2.warpPerspective(listaImages[2], np.linalg.inv(np.dot(H[0], H[1])), (ancho * 2, alto)) #trasformada de 2 a 0

        listaImagesTrasformadas.append(image_warped)#agrega a lista

        return listaImagesTrasformadas #retorna lista
    if pos == 1: #Stitchin de izq----> derecha y luego de derecha --->izq
        image_warped = cv2.warpPerspective(listaImages[0], H[0], (ancho * 2, alto))#image 0 trasformada
        listaImagesTrasformadas.append(image_warped)

        image_warped = cv2.warpPerspective(listaImages[2], np.linalg.inv(H[1]), (ancho * 2, alto))# image 2 trasformada a ref
        image_warped[0:listaImages[1].shape[0], 0:listaImages[1].shape[1]] = listaImages[1]#pega la ref y 2 trasformada

        listaImagesTrasformadas.append(image_warped)

        return listaImagesTrasformadas# retorna img 1 trasformada y la union de ref con la 2 trasformada
    if pos == 2:
        imgzeros = np.zeros((listaImages[0].shape[0], listaImages[0].shape[1], 3), np.uint8)
        image_warped1 = cv2.warpPerspective(listaImages[0], np.dot(H[0], H[1]), (ancho * 2, alto)) #trasforma de 0 hacia 2
        cv2.imshow("imagen Warped 0 a 2", image_warped1)
        cv2.waitKey(0)
        image_warped = cv2.warpPerspective(listaImages[1], H[1], (ancho*2, alto))# trasforma de 1 hacia 2

        cv2.imshow("imagen Warped 1 a 2", image_warped)
        cv2.waitKey(0)

