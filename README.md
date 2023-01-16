# Artificial Intelligence to Solve Captchas

This is an artificial intelligence trained using the deep learning neural network method to identify captcha images of letters and numbers and return the value in text to the user.
This project was developed exclusively for the purpose of studying topics related to:
- Artificial Intelligence with Python
- Machine Learning
- Deep Learning Neural Networks

</br>

## :heavy_check_mark: Project Status
:white_check_mark: Project Finished! :white_check_mark:

</br>

## ⚙️ How It Works
First, the captcha image is treated to reduce all imperfections and make the letters as clear as possible - with a clean image of the letters, A.I. can identify pixel fill patterns.

After the image is clean, it goes to the next step: recognizing the areas with the most fill so that the parts with the most pixels are recognized as letters. The parts with the most dark pixels are outlined - this way we can separate the letters and send them one at a time to A.I. to predict their value.

After these steps, we just need to pass each image to the A.I. to predict its value (remembering that the A.I. has already been trained and its model is saved in the repository folders).

</br>

## 🤖 A.I.
The A.I. training module is in the "AI_Training" folder. The deep learning neural network method was used for training.

**The achieved accuracy of the A.I. was 87%.**

Considering that the project's purpose was to study the concepts of Artificial Intelligence, a large amount of data was not used for training. If the training was done with a larger amount of data, the accuracy of the AI would be higher!

</br>

## ☕ How to Use?
To use/test the captcha solver A.I. is simple. Just import the file "captcha_solver.py" into your project and use the function "solve_captcha()".

By default, the A.I. will read the images from the "captcha_in" folder and after processing the image, it will be placed in the "captcha_out" folder. 

❗**Important**: If the folders are not created, A.I. will search unsuccessfully for the folders and the program will fail. Make sure that these folders are created with their proper names.

Also, remember to install all the modules that A.I. uses

</br>

## 🛠️ Developed With
PyCharm 2022.1.3 (Community Edition)
Build #PC-221.5921.27, built on June 21, 2022
Runtime version: 11.0.15+10-b2043.56 amd64
VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o.
Windows 11 10.0
GC: G1 Young Generation, G1 Old Generation
Memory: 942M
Cores: 12

## 📝 License
This project is under license. See the file [LICENÇA](LICENSE) for more details.

[⬆ Back to the top](https://github.com/KokumaiLuis/artificial_intelligence_captcha_solver)<br>
