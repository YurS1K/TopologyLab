import cv2
from numpy import hypot

BG = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# Создание маски для определения контуров и её зачистка
def create_mask(frame_for_mask):
    mask = BG.apply(frame_for_mask, learningRate=0.004)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)

    # Убираем из маски область, где едет поезд
    mask[0:140, 0:1280] = 0
    cv2.imshow("mask", mask)
    return mask


# Получение центроида
def get_centroid(x, y, w, h):
    x_centroid = int(x + w/2)
    y_centroid = int(y + h/2)

    return [x_centroid, y_centroid]


# Размещение текста в кадре
def put_text(frame_for_edit, k, s):
    cv2.putText(frame_for_edit, f'Active Cars: {k}', (50, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Avg Speed: {s} px/s', (50, 150),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return frame_for_edit


# Чтение видео
cap = cv2.VideoCapture('video.mp4')

prev_centroids = []

# Получение количества fps видео
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    # Извлекаем кадр для обработки
    ret, frame = cap.read()

    # Точка остановки (конец видео)
    if not ret:
        break

    # Создание маски
    mask = create_mask(frame)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Фильтрация мелких контуров
    min_contour_area = 600
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Расчёт центроидов, рисование рамки
    cur_centroids = []

    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cur_centroids.append(get_centroid(x, y, w, h))

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h ), (0, 255, 0), 2)

    # Расчет скорости
    speeds = []
    if prev_centroids:
        for cur in cur_centroids:

            # Расчет дистанций между объекта и другими объектами
            distances = []
            for prev in prev_centroids:
                distances.append(hypot(cur[0] - prev[0], cur[1] - prev[1]))

            # Так как порядок, в котором записываются центроиды в список, хаотичен
            # Нужно сопоставить объект в текущем кадре с собой в предыдущем кадре
            # Делаем это с помощью нахождения минимальной дистанции и если она допустима, то рассчитываем скорость объекта
            min_dist = min(distances) if distances else 0
            if min_dist < 50:
                # Заводим список скоростей, чтобы позже рассчитать среднюю скорость объектов
                speeds.append(min_dist / (1/fps))

    # Средняя скорость объектов
    avg = sum(speeds) / len(speeds) if len(speeds) else 0

    # Переводим обработанные центроиды в предыдущие
    prev_centroids = cur_centroids

    # Вывод информации о количестве движущихся машин в кадре и их суммарной средней скорости
    cv2.imshow("Video", put_text(frame, len(speeds), int(avg)))

    if cv2.waitKey(27) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()