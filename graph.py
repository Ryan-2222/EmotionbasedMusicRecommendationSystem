from matplotlib import pyplot as plt

with open("./model/model_fit_log", "r") as f:
    data = eval(f.read())

# Accuracy Graph
plt.plot(data['accuracy'])
plt.plot(data['val_accuracy'])
plt.title('data accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Loss Graph
plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.title('data loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()