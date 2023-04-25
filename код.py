# Importing Model libraries
import cv2  # Install opencv-python
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
import tensorflow
# Импорт библиотек бота
import telebot

# Подключение токена. Файл TOKEN содержит токен данные для подключения к боту.
from datatoken2 import TOKEN
bot = telebot.TeleBot(TOKEN)
# print(TOKEN)

#Creating and opening a file joined.txt to write to it the user id of the bot
joinedFile = open("joined.txt", "r")
joinedUsers = set ()
for line in joinedFile:
   joinedUsers.add(line.strip())
joinedFile.close()

# When logging into the bot, the user sends the /start command.
# After that, the user ID is recorded in a file for further use during mailing.
@bot.message_handler(commands=['start'])
def startjoin(message):
   if not str(message.chat.id) in joinedUsers:
      joinedFile = open("joined.txt", "a")
      joinedFile.write(str(message.chat.id) + "\n")
      joinedUsers.add(message.chat.id)

# Launch of the weapon detection model on the video stream. The model is trained through the website Teachable Machine.

######
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

    # itog_model a variable that gets the name of the result of the model (weapon или mask)
    itog_model = str(class_name[2:])

    weapon = '''weapon
'''

    # proc a variable that gets the number of percentages of the result of the model
    proc = int(confidence_score * 100+1)

    # If in the video the model determines the weapon value and the percentage of match is greater than 70,
    # then the bot users receive a notification.
    # Depending on the distance from the intended objects to the camera, the percentage of coincidence can be changed in the condition.
    if proc > 90 and itog_model == weapon:
        print('Человек с оружием')
        for user in joinedUsers:
            bot.send_message(user, 'Внимание! Человек c оружием!')


bot.polling()

camera.release()
cv2.destroyAllWindows()
#####
