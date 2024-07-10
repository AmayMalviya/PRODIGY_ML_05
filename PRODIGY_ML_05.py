import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import cv2

# Directory paths
dataset_dir = '/Users/amaymalviya/Downloads/food_dataset/Food 101'  # Update with the actual path to your dataset

# Image parameters
img_width, img_height = 224, 224

# Function to load and preprocess images and calories
def load_images_and_calories(dataset_dir, img_width, img_height):
    images = []
    labels = []
    calories = []
    label_map = {}
    label_counter = 0

    for food_item in os.listdir(dataset_dir):
        food_item_dir = os.path.join(dataset_dir, food_item)
        if os.path.isdir(food_item_dir):
            if food_item not in label_map:
                label_map[food_item] = label_counter
                label_counter += 1
            label = label_map[food_item]
            
            # Assume calorie information is stored in a file named 'calories.txt' in the food_item_dir
            with open(os.path.join(food_item_dir, 'calories.txt'), 'r') as f:
                calorie_content = int(f.read().strip())

            for img_file in os.listdir(food_item_dir):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(food_item_dir, img_file)
                    img = load_img(img_path, target_size=(img_width, img_height))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                    calories.append(calorie_content)

    return np.array(images), np.array(labels), np.array(calories), label_map

# Load and preprocess images and calories
images, labels, calories, label_map = load_images_and_calories(dataset_dir, img_width, img_height)

# Preprocess images for MobileNetV2
images = preprocess_input(images)

# Convert labels to categorical
labels = to_categorical(labels, num_classes=len(label_map))

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(label_map), activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val, cal_train, cal_val = train_test_split(images, labels, calories, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Function to estimate calories
def estimate_calories(predictions, label_map, calorie_map):
    predicted_labels = np.argmax(predictions, axis=1)
    estimated_calories = [calorie_map[list(label_map.keys())[label]] for label in predicted_labels]
    return estimated_calories

# Predict on the validation set
predictions = model.predict(X_val)

# Create a calorie map
calorie_map = {food: calorie for food, calorie in zip(label_map.keys(), cal_train)}

# Estimate calories
estimated_calories = estimate_calories(predictions, label_map, calorie_map)

# Compare estimated calories with actual calories
print(f'Actual Calories: {cal_val[:10]}')
print(f'Estimated Calories: {estimated_calories[:10]}')

# Evaluate classification performance
y_val_pred = np.argmax(predictions, axis=1)
y_val_true = np.argmax(y_val, axis=1)
print(classification_report(y_val_true, y_val_pred, target_names=list(label_map.keys())))

# Evaluate calorie estimation
calorie_accuracy = accuracy_score(cal_val, estimated_calories)
print(f'Calorie Estimation Accuracy: {calorie_accuracy}')

# Function to preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_width, img_height))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Real-time food recognition and calorie estimation
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = preprocess_frame(frame_rgb)
    predictions = model.predict(processed_frame)
    predicted_label = np.argmax(predictions, axis=1)[0]
    food_item = list(label_map.keys())[predicted_label]
    estimated_calories = calorie_map[food_item]

    # Display the food item and calorie content on the frame
    cv2.putText(frame, f'{food_item}: {estimated_calories} kcal', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Food Recognition and Calorie Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
