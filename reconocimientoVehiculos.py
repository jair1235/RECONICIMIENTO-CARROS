import cv2

# video
capture = cv2.VideoCapture("videoPrueba1.mov")

# Clasificador Haar
carros = cv2.CascadeClassifier("cars.xml")

# guia verde
line_y = 300

while True:
    ret, frames = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = carros.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x, y, w, h) in cars:
        # Dibujar rectángulo sobre cada objeto detectado
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Dibujar la línea verde
    cv2.line(frames, (0, line_y), (frames.shape[1], line_y), (0,255,0), 2)

    cv2.imshow("Detección de autos y motos", frames)
    if cv2.waitKey(33) == 27:  # ESC para salir
        break

capture.release()
cv2.destroyAllWindows()