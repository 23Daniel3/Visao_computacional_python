from ultralytics import YOLO
import time

modelo = YOLO('yolov8n.pt')

last_detection_time = 0
detection_interval = 3  

results = modelo.predict(source='0', save=False, show=True, stream=True)

for result in results:
    mensagem = False
    current_time = time.time()
    
    for box in result.boxes:
        if box.cls == 0:
            mensagem = True
    
    if mensagem and (current_time - last_detection_time >= detection_interval):
        print("Pessoa detectada!")
        last_detection_time = current_time
