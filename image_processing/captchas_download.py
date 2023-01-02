import pyautogui as pa
import time

for c in range(4, 204):
    time.sleep(3)
    pa.press('f5')
    time.sleep(1)
    pa.moveTo(960, 405)
    pa.rightClick()
    pa.move(2, 73)
    pa.leftClick()
    time.sleep(3)
    pa.write(f'captcha{c}')
    pa.press('enter')
