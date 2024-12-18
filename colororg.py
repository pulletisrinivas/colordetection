!pip install opencv-python-headless matplotlib ipywidgets tensorflow keras

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


img_path = "/content/colorpic.jpg"
csv_path = "/content/colors.csv"


img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(csv_path, names=index, header=None)


X = csv[['R', 'G', 'B']].values
y = csv['color_name'].values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)


X_train = X_train.reshape(-1, 3, 1)
X_test = X_test.reshape(-1, 3, 1)


num_classes = y_categorical.shape[1]
model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(3, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
)


model.fit(
    X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test)
)


def on_button_click(b):
    x = int(x_coord.value)
    y = int(y_coord.value)

    r_val, g_val, b_val = img[y, x]
    r_val, g_val, b_val = int(r_val), int(g_val), int(b_val)

   
    input_rgb = np.array([[r_val, g_val, b_val]]).reshape(-1, 3, 1)
    pred = model.predict(input_rgb)
    pred_label = np.argmax(pred, axis=1)
    color_name = label_encoder.inverse_transform(pred_label)[0]

    
    img_with_info = img.copy()
    cv2.rectangle(img_with_info, (20, 20), (750, 60), (r_val, g_val, b_val), -1)
    plt.imshow(img_with_info)
    plt.title(f"Selected Color: {color_name} - R={r_val}, G={g_val}, B={b_val}")
    plt.axis("off")
    plt.show()


plt.imshow(img)
plt.title("Color Picker - Original Image")
plt.axis("on")
plt.show()

x_coord = widgets.IntText(value=0, description='X Coordinate:')
y_coord = widgets.IntText(value=0, description='Y Coordinate:')
button = widgets.Button(description="Get Color")
button.on_click(on_button_click)

display(x_coord, y_coord, button)
