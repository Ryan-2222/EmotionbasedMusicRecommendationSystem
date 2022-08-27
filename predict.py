import cv2
import numpy as np
from keras.models import model_from_json
import webbrowser
import random

model_path = './model/'
img_size = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
song_path = "./songs/"
num_class = len(emotion_labels)

json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(model_path + 'model_weight.h5')

capture = cv2.VideoCapture(1)

cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')

emotional_list = []

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=1, minSize=(120, 120))

    if len(faceLands) > 0:
        for faceLand in faceLands:
            x, y, w, h = faceLand
            images = []
            result = np.array([0.0] * num_class)

            image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
            image = image / 255.0
            image = image.reshape(1, img_size, img_size, 1)

            predict_lists = model.predict(image, batch_size=32, verbose=1)
            # print(predict_lists)
            result += np.array([predict for predict_list in predict_lists
                                for predict in predict_list])
             # print(result)
            emotion = emotion_labels[int(np.argmax(result))]
            print("Emotion:", emotion)


            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20),
                          (0, 255, 255), thickness=10)
            cv2.putText(frame, '%s' % emotion, (x, y - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2, 30)
            cv2.imshow('Face', frame)


        emotional_list.append(emotion)

        # Songs Selection
        if len(emotional_list) >= 100:
            print(f"Final Emotion Detected: {max(set(emotional_list), key=emotional_list.count)}")
            with open(f'{song_path}{max(set(emotional_list), key=emotional_list.count)}.txt') as f:
                lines = f.readlines()
                webbrowser.open(random.choice(lines))
            if emotional_list[-1] == "angry":
                pass
            break

        if cv2.waitKey(60) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()