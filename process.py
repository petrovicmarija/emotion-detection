import os
import numpy as np
import cv2
import argparse
import imutils
import dlib
from imutils import face_utils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

NBINS = 9
CELL_SIZE = (8, 8)
BLOCK_SIZE = (3, 3)


def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran. 
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu). 
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran

    images = []
    for path in train_image_paths:
        img = resize_image(load_image(path))
        images.append(img)

    print(len(images))

    features = []
    labels = []

    hog = get_hog(img)

    for i in range(len(images) - len(images) + 1):
        for j in range(len(train_image_labels)):
            features.append(hog.compute(images[j]))
            labels.append(train_image_labels[j])

    features = np.array(features, dtype="object")
    y = np.array(labels, dtype="object")

    x_train = reshape_data(features)
    print(x_train.shape)
    print('Train shape: ', x_train.shape, y.shape)

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x_train, y)
    y_train_pred = clf_svm.predict(x_train)
    print("Train accuracy: ", accuracy_score(y, y_train_pred))

    model = clf_svm
    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """

    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    facial_expression = ""
    # possible_facial_expressions = ["anger", "contempt", "disgust", "happiness", "neutral", "sadness", "surprise"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = load_image(image_path)
    print(image.shape)
    image = cv2.resize(image, (250, 250))
    rects = detector(image, 1)

    shape = []
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        print("Dimenzije prediktor matrice: {0}".format(shape.shape))
        print("Prva 3 elementa matrice")
        print(shape[:3])

    #display_image(image)
    hog = get_hog(image)

    feature = hog.compute(image)
    feature = reshape_data_extract(feature)

    # if len(shape) == 0:
    #     facial_expression = 'anger'
    # else:
    facial_expression = trained_model.predict(feature)[0]

    return facial_expression


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def resize_image(image):
    resized = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
    return resized


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def reshape_data_extract(data):
    nx, ny = data.shape
    return data.reshape((1, nx*ny))


def get_hog(image):
    return cv2.HOGDescriptor(_winSize=(image.shape[1] // CELL_SIZE[1] * CELL_SIZE[1],
                                       image.shape[0] // CELL_SIZE[0] * CELL_SIZE[0]),
                             _blockSize=(BLOCK_SIZE[1] * CELL_SIZE[1],
                                         BLOCK_SIZE[0] * CELL_SIZE[0]),
                             _blockStride=(CELL_SIZE[1], CELL_SIZE[0]),
                             _cellSize=(CELL_SIZE[1], CELL_SIZE[0]),
                             _nbins=NBINS)
