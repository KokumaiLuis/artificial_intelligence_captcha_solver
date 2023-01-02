import cv2
import os
import glob

files = glob.glob('image_processing/cleaned_captchas/*')
for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 243, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    letters_region = []

    for contour in contours:
        (x, y, width, height) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            letters_region.append((x, y, width, height))

    final_image = cv2.merge([img] * 3)

    for index, rectangle in enumerate(letters_region):
        if index == 0:
            continue
        x, y, width, height = rectangle
        letter_img = img[y-2:y+height+2, x-2:x+width+2]
        file_name = os.path.basename(file).replace(".png", f"letter{index}.png")
        cv2.imwrite(f'image_processing/captcha_letters/{file_name}', letter_img)
        cv2.rectangle(final_image, (x - 2, y - 2), (x + width + 2, y + height + 2), (0, 255, 0), 1)
    file_name = os.path.basename(file)
    cv2.imwrite(f"image_processing/localized_letters/{file_name}", final_image)
