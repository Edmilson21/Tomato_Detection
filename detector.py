from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import colors


# Cargar el modelo entrenado
try:
    model = YOLO('') # Ruta de tu modelo , ej: ('model.pt')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

#ip = 'https://192.168.169.132:8080/video'
cap = cv2.VideoCapture("#") # Ruta de tu video

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

tamaño=(500,600)

while cap.isOpened():
    #success, im0 = cap.read()
    ret, frame = cap.read()

    # redimencionar la pantalla
    frame = cv2.resize(frame,tamaño)
    if ret != True:
        print("Hay un problema con su cámara")
        break

    # Se hace la predicción y el rastreamiento
    results = model(frame)

    # Extraer las cajas delimitadoras, muestra las clases, ID, y la confianza de los resultados
    boxes = results[0].boxes.xyxy.cpu().tolist() # Coordenadas de las cajas
    clss = results[0].boxes.cls.cpu().tolist()   # Clase de la detección
    confs = results[0].boxes.conf.cpu().tolist() # Confianza de las detecciones

    # Iterando sobre todas las detecciones para dibujar la caja delimitadora
    if boxes is not None:
        for box, cls, conf in zip(boxes, clss, confs):
            # Definiendo el color de la caja, con base en la clase
            color = colors(int(cls), True)

            # Dibujar la Caja Delimitadora
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            # Diseñando el rótulo con la clase, confianza e ID de rastreamiento
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el fotograma
    cv2.imshow("Salida", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()