from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from captcha_cleaner import clean_images


def quebrar_captcha():
    # importar o modelo que a gente treinou e importar o tradutor
    with open("AI_training/labels_model.dat", "rb") as translate_file:
        lb = pickle.load(translate_file)

    model = load_model("AI_training/trained_model.hdf5")

    # usar o modelo pra resolver os catpchas
    clean_images("captcha_in", out_path="captcha_out")
    # ler todos os arquivos da pasta "resolver"
    ######
    files = list(paths.list_images("captcha_out"))
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # em preto e branco
        _, img = cv2.threshold(img, 243, 255, cv2.THRESH_BINARY_INV)

        # encontrar os contornos de cada letra
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        letters_region = []

        # filtrar os contornos que são realmente de letras
        for contour in contours:
            (x, y, width, height) = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 100:
                letters_region.append((x, y, width, height))

        letters_region = sorted(letters_region, key=lambda x: x[0])
        # desenhar os contornos e separar as letras em arquivos individuais
        final_image = cv2.merge([img] * 3)
        prediction = []

        for rectangle in letters_region:
            x, y, width, height = rectangle
            letter_img = img[y-2:y+height+2, x-2:x+width+2]

            # dar a letra pra inteligencia artificial descobrir que letra é essa
            letter_img = resize_to_fit(letter_img, 20, 20)

            # tratamento para o Keras funcionar
            letter_img = np.expand_dims(letter_img, axis=2)
            letter_img = np.expand_dims(letter_img, axis=0)

            predicted_letter = model.predict(letter_img)
            predicted_letter = lb.inverse_transform(predicted_letter)[0]
            prediction.append(predicted_letter)

        predicted_text = "".join(prediction)
        print(predicted_text)
        return predicted_text


if __name__ == "__main__":
    quebrar_captcha()