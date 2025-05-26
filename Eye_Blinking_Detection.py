import cv2
import numpy as np
import dlib
from math import hypot
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk

# Usamos el detector para detectar el rostro frontal
detector = dlib.get_frontal_face_detector()

# Detecta los puntos de referencia faciales
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

# Creamos una función que necesitaremos más adelante para detectar el punto medio.
# En esta función simplemente ponemos las coordenadas de dos puntos y devolverá el punto medio
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks, frame):
    # Detecta el lado izquierdo del ojo
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    # Detecta el lado derecho del ojo
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # Detecta el punto medio superior del ojo
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    # Detecta el punto medio inferior del ojo
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # Dibuja la línea horizontal del ojo
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # Dibuja la línea vertical del ojo
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    # Calcula la distancia de la línea horizontal
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    # Calcula la distancia de la línea vertical
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    # Calcula el ratio
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    # Extrae la región del ojo
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    h, w = threshold_eye.shape

    left_side = threshold_eye[0:h, 0:int(w / 2)]
    left_white = cv2.countNonZero(left_side)

    right_side = threshold_eye[0:h, int(w / 2):w]
    right_white = cv2.countNonZero(right_side)

    if left_white == 0:
        gaze_ratio = 1
    elif right_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_white / right_white
    return gaze_ratio

cap = cv2.VideoCapture(0)
blink_counter = 0
last_blink_time = 0

screen_width, screen_height = pyautogui.size()

def actualizar_frame():
    ret, frame = cap.read()

    # Validar que el frame fue capturado correctamente
    if not ret or frame is None:
        print("❌ No se pudo capturar el frame.")
        root.after(10, actualizar_frame)
        return

    # Validar que el frame tiene el tipo correcto
    if frame.dtype != 'uint8':
        frame = frame.astype('uint8')

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Validar tipo de gray_image
    if gray_image.dtype != 'uint8':
        gray_image = gray_image.astype('uint8')

    # Ahora puedes detectar sin problemas
    faces = detector(gray_image)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray_image, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks, frame)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks, frame)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Detección de parpadeo y clic
        if blinking_ratio > 5.7:
            cv2.putText(frame, "PARPADEO", (50, 150), font, 7, (255, 0, 0))
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_blink_time > 1:  # Evita múltiples clics por parpadeo
                pyautogui.click()
                last_blink_time = current_time

        # Seguimiento de la mirada
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray_image)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray_image)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        # Determina la dirección de la mirada y mueve el mouse
        if gaze_ratio <= 0.8:
            cv2.putText(frame, "MIRANDO A LA DERECHA", (50, 100), font, 3, (0, 0, 255))
            pyautogui.moveRel(40, 0)
        elif gaze_ratio > 1.8:
            cv2.putText(frame, "MIRANDO A LA IZQUIERDA", (50, 100), font, 3, (0, 0, 255))
            pyautogui.moveRel(-40, 0)
        else:
            cv2.putText(frame, "MIRANDO AL CENTRO", (50, 100), font, 3, (0, 0, 255))

    # Convertir el frame a formato compatible con Tkinter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)
    root.after(10, actualizar_frame)

def cerrar():
    cap.release()
    root.destroy()

root = tk.Tk()
root.title("Detección de rostros (GUI)")
lbl_video = tk.Label(root)
lbl_video.pack()
btn_salir = tk.Button(root, text="Salir", command=cerrar)
btn_salir.pack()

actualizar_frame()
root.mainloop()