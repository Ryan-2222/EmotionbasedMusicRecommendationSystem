# Emotion based Music Recommendation

Recommend you a piece of music that suits you based on your current mood!

## Hardware Requirements
- A suitable camera installed 
- A PC

## Libraries Required
- Tensorflow
- Keras
- opencv-python
- pillow
- matplotlib
- scipy

```python
pip install tensorflow
pip install keras
pip install opencv-python
pip install pillow
pip install matplotlib
pip install scipy
```

## How to use?
- Open cmd in windows.
- Run the program with the following command in cmd.

```python
py predict.py
```

After running for a while, you can see your face in your screen which like the picture has shown below.
(The program only starts running when your face is detected).

![avatar](images/sample.png)

Enjoy your music time!

###### Notice: Music can be edited on ```songs/{emotion}.txt``` if you would like to.

---
# Face Emotion Recognizer (For AI Model Testing)
If you would like to try the AI model only, you may run the following command in cmd.

## Library Required

- streamlit
- streamlit_webrtc
```python
pip install streamlit
pip install streamlit_webrtc
```

## How to use?
- Open cmd and run the following command
```python
py app_run.py
```

---