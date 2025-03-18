import os
import cv2 as cv
import numpy as np
import winsound  # Для воспроизведения звука в Windows


def detect(selected: list = None) -> None:
    # Проверка наличия файлов
    if not os.path.exists("yolov3.weights"):
        print("Error: yolov3.weights file not found")
        exit()
    if not os.path.exists("yolov3.cfg"):
        print("Error: yolov3.cfg file not found")
        exit()
    if not os.path.exists("coco.names"):
        print("Error: coco.names file not found")
        exit()

    # Загрузка модели YOLO
    net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Открытие веб-камеры
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    # Установка разрешения
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # Задаем зону координат
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Координаты зоны срабатывания
    zone_x1 = int(frame_width * 0.02)  # left
    zone_y1 = int(frame_height * 0.98)  # top
    zone_x2 = int(frame_width * 0.98)  # right
    zone_y2 = int(frame_height * 0.02)  # bottom

    # Создание директории для сохранения скриншотов
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            break

        height, width, channels = frame.shape

        # Подготовка изображения для детекции
        blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Обработка детекций
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Подавление не максимальных значений
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Нарисование ограничивающих рамок и зоны
        font = cv.FONT_HERSHEY_PLAIN
        is_detected = False  # Флаг для отслеживания обнаружения чашки
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # Обычная рамка для всех объектов
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, label, (x, y - 10), font, 1, (0, 255, 0), 2)

                # Особая обработка для выбранных объектов
                if label in selected and selected is not None:
                    is_detected = True
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красная рамка
                    cv.putText(frame, label, (x, y - 10), font, 1, (0, 0, 255), 2)
                    print("It's a ", label, " at coordinates:", x, y, w, h)
                    save_path = f"screenshots/detected_{x}_{y}_{w}_{h}.jpg"
                    resized_frame = cv.resize(frame, (640, 480))
                    cv.imwrite(save_path, resized_frame)
                    print(f"Saved image: {save_path}")

                # # Проверка пересечения с зоной и сохранение изображения
                # if not (x > zone_x2 or x + w < zone_x1 or y > zone_y2 or y + h < zone_y1) and is_detected:
                #     save_path = f"screenshots/detected_{x}_{y}_{w}_{h}.jpg"
                #     resized_frame = cv.resize(frame, (224, 224))
                #     cv.imwrite(save_path, resized_frame)
                #     print(f"Saved image: {save_path}")
                # else:
                #     print(f"Object not in zone: {x}, {y}, {w}, {h}")

        # Если обнаружена чашка, издаем звук
        if is_detected:
            winsound.Beep(500, 200)  # Частота 500 Гц, длительность 200 мс

        # Отрисовка зоны поиска
        cv.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)

        # Отображение текущего кадра
        cv.imshow("Webcam", frame)

        # Выход по нажатию 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv.destroyAllWindows()


detect(['cup', 'cell phone'])
