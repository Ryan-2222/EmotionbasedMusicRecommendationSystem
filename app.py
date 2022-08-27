import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
from matplotlib import pyplot as plt
import queue
import random
from streamlit_player import st_player

with open("./model/model_fit_log", "r") as f:
    data = eval(f.read())

model_path = './model/'
img_size = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
song_path = "./songs/"
pixel_format = "bgr24"
num_class = len(emotion_labels)

json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(model_path + 'model_weight.h5')
cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_default.xml')
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

emotion_list = []
# emotion_list = emotion_list[:101]
proba = queue.Queue()

def video_frame_callback(frame):
    frm = frame.to_ndarray(format=pixel_format)
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
    if len(faceLands) > 0:
        for faceLand in faceLands:
            x, y, w, h = faceLand
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
            emotion_list.append(emotion)

            proba.put(predict_lists)

            # Proba in Camera
            # for i, label in enumerate(emotion_labels):
            #     cv2.putText(frm, f"{label}: {round(predict_lists[0][i]*100, 2)}", (0, 17+20*i), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, 30)

            # print("Emotion:", emotion)
            cv2.rectangle(frm, (x - 20, y - 20), (x + w + 20, y + h + 20),
                          (0, 255, 255), thickness=10)
            cv2.putText(frm, '%s' % emotion, (x, y - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2, 30)

    return av.VideoFrame.from_ndarray(frm, format=pixel_format)


st.set_page_config(
    page_title="Emotion based Music Recommendation",
    page_icon="🎧",
    layout="centered"
)

def main():
    st.title("Emotion Based Music Recommendation 🎧")
    stop_warn = st.empty()
    ctx = webrtc_streamer(key="youmustnotpasslol", video_frame_callback=video_frame_callback, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"audio": False, "video": True})
    st.markdown("###### Let's try to make some emotions!")

    # for loop method failed
    title = st.sidebar.empty()
    angry = st.sidebar.empty()
    disgust = st.sidebar.empty()
    fear = st.sidebar.empty()
    happy = st.sidebar.empty()
    sad = st.sidebar.empty()
    surprise = st.sidebar.empty()
    neutral = st.sidebar.empty()

    while ctx.state.playing:
        if len(emotion_list) <= 100:
            with st.sidebar.container():
                title.markdown("## Probability")
                angry.text(f"angry: {round(proba.get()[0][0]*100, 2)}%")
                disgust.text(f"disgust: {round(proba.get()[0][1]*100, 2)}%")
                fear.text(f"fear: {round(proba.get()[0][2]*100, 2)}%")
                happy.text(f"happy: {round(proba.get()[0][3]*100, 2)}%")
                sad.text(f"sad: {round(proba.get()[0][4]*100, 2)}%")
                surprise.text(f"surprise: {round(proba.get()[0][5]*100, 2)}%")
                neutral.text(f"neutral: {round(proba.get()[0][6]*100, 2)}%")
        else:
            st.write(f"Final Emotion Detected: {max(set(emotion_list), key=emotion_list.count)}")
            # Songs Selection
            with open(f'{song_path}{max(set(emotion_list), key=emotion_list.count)}.txt') as f:
                lines = f.readlines()
            st_player(random.choice(lines))
            stop_warn.warning("Please press the STOP button to restart the application")
            break


def Graph_page():
    # Graph
    st.title("Data Accuracy and Loss Graph")
    left, right = st.columns(2)
    with left:
        try:
            plt.clf()
            # Accuracy Graph
            plt.plot(data['accuracy'])
            plt.plot(data['val_accuracy'])
            plt.title('Data Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            st.pyplot(plt)
        except:
            st.write("Please Rerun (Press R) to re-generate the graph")

    with right:
        try:
            plt.clf()
            # Loss Graph
            plt.plot(data['loss'])
            plt.plot(data['val_loss'])
            plt.title('Data Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            st.pyplot(plt)
        except:
            st.write("Please Rerun (Press R) to re-generate the graph")

    st.markdown("---")

page_names_to_func = {
    "Emotion based Music Recommendation 🎧": main,
    "AI Model Data Graph  📈": Graph_page
}

st.sidebar.title("Welcome!")
selected_page = st.sidebar.selectbox("Select a page", page_names_to_func.keys())
page_names_to_func[selected_page]()
st.sidebar.markdown("### Our team: CMM GUNDAM")

st.sidebar.write("#### Developers:")
st.sidebar.write("Leaf Gear")
st.sidebar.write("Ryan_2222")
st.sidebar.write("helloiamepicccccc")
st.sidebar.write("最愛亞絲娜")
st.sidebar.write("M&G")