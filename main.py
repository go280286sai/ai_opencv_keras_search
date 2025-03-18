import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.utils import load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# 1. Подготовка генератора данных
def image_generator(file_paths, labels, batch_size=32, target_size=(224, 224)):
    while True:
        for start in range(0, len(file_paths), batch_size):
            end = min(start + batch_size, len(file_paths))
            batch_paths = file_paths[start:end]
            batch_labels = labels[start:end]
            images = []
            for img_path in batch_paths:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
            yield np.array(images), np.array(batch_labels)


# Пути к папкам с изображениями
crash_folder = "img/crash"
no_crash_folder = "img/no_crash"

# Список файлов и меток
crash_files = [os.path.join(crash_folder, filename) for filename in os.listdir(crash_folder)]
no_crash_files = [os.path.join(no_crash_folder, filename) for filename in os.listdir(no_crash_folder)]

# Объединяем данные и метки
file_paths = crash_files + no_crash_files
labels = [1] * len(crash_files) + [0] * len(no_crash_files)

# Разделяем на тренировочную и тестовую выборки
file_paths_train, file_paths_test, labels_train, labels_test = train_test_split(file_paths, labels, test_size=0.2,
                                                                                random_state=42)

# Размер пакета
batch_size = 32

# 2. Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Обновляем input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 3. Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Обучение модели с использованием генератора
train_generator = image_generator(file_paths_train, labels_train, batch_size)
validation_generator = image_generator(file_paths_test, labels_test, batch_size)
steps_per_epoch = len(file_paths_train) // batch_size
validation_steps = len(file_paths_test) // batch_size

history = model.fit(train_generator, epochs=3, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator, validation_steps=validation_steps)

# 5. Оценка модели
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_steps)
print(f"Точность на тестовой выборке: {test_acc}")


# 6. Предсказание
def predict_image(image_path, model, target_size=(224, 224)):  # Обновляем target_size
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 1 if prediction[0] > 0.5 else 0


# Пример использования

import cv2 as cv


def find_event(url: str):
    cap = cv.VideoCapture(url)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Если кадры закончились
            break

        # Сохранение текущего кадра как временное изображение (необязательно)
        temp_image_path = 'temp_frame.png'
        cv.imwrite(temp_image_path, frame)

        result = predict_image(temp_image_path, model)
        if result == 1:
            cap.release()
            cv.destroyAllWindows()
            return 1

        # Нажмите 'q' для выхода из окна отображения
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return 0

print(find_event('data/00000.mp4'))
# data_train = pd.read_csv("data/test.csv")
# data_train = data_train[['id']]
# result = []
# for el in data_train['id']:
#     url = f"data/test/{str(el).rjust(5, '0')}.mp4"
#     prediction = find_event(url)
#     result.append(prediction)
#     print(f"ID: {el}, Предсказание: {prediction}")
# params = {'id': data_train['id'], 'result': pd.Series(result)}
#
# df = pd.DataFrame(params)
# print(df)
# df.to_csv("prediction.csv", index=False)
