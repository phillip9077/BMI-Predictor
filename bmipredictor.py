import pandas as pd
import numpy as np
from sklearn import model_selection
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential

# reading data from the .csv file
# usecols is a parameter that allows us to specify which columns of data we want
dataframe = pd.read_csv('C:/VIA Tech/BMI Classifier/vip/annotation.csv',
                        usecols=['image', 'BMI']).values
X = []  # the features, or inputs, 
y = []  # the labels, or outputs

# separating the inputs and the outputs, aka our images and their corresponding BMI values
for row in dataframe:
    image_name = row[0] + '.jpg'
    image_bmi = row[1]
    # resizing our original images to 256x256, and turning them into numpy arrays
    image = load_img('C:/VIA Tech/BMI Classifier/vip/data/' + image_name, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array(input_arr)
    X.append(input_arr)
    y.append(image_bmi)

X = np.array(X)
y = np.array(y)

# obtaining the training and testing dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# The setup for a convolutional neural network (CNN)
model = Sequential()
model.add(keras.Input(shape=(128, 128, 3)))
model.add(keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=8, kernel_size=5, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))

# Training the CNN with our augmented training dataset
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data_gen.flow(X_train, y_train, batch_size=16), epochs=8)

# Saving the model to allow for faster testing in the future
model.save('C:/VIA Tech/BMI Classifier')

# Once you trained the model once, you can comment out the above and uncomment the following line
# model = keras.models.load_model('C:/VIA Tech/BMI Classifier')

# Testing the CNN with our testing dataset
results = model.evaluate(X_test, y_test, verbose=0)
print('Mean squared error = ', results)

# Testing the CNN with your own picture
img = load_img('C:/VIA Tech/BMI Classifier/me.jpg', target_size=(128, 128))
img_array = np.array(img_to_array(img))
img_array = img_array.reshape(1, 128, 128, 3)
img_array = img_array.astype('float32') / 255

prediction = model.predict(img_array)
print('BMI prediction is:', prediction)