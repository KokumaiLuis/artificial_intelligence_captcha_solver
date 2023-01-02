import cv2
import os
import glob


def clean_images(in_path, out_path='image_processing/cleaned_captchas'):
    files = glob.glob(f'{in_path}/*')
    for file in files:
        img = cv2.imread(file)
        cleaned_img = cv2.medianBlur(img, 5)
        #cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2GRAY)
        _, cleaned_img = cv2.threshold(cleaned_img, 127, 255, cv2.THRESH_TOZERO)
        #cleaned_img = cv2.Laplacian(cleaned_img, cv2.CV_8UC1)

        file_name = os.path.basename(file)
        cv2.imwrite(f'{out_path}/{file_name}', cleaned_img)


if __name__ == '__main__':
    clean_images('image_processing/captcha_dataset')
