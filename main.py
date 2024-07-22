from ultralytics import YOLO
import time
import winsound
import keyboard
import threading

def play_sound():
    while playing_sound:
        winsound.Beep(1000, 500)

playing_sound = False

modelo = YOLO('yolov8n.pt')

last_detection_time = 0
detection_interval = 3  

results = modelo.predict(source='0', save=False, show=True, stream=True)

for result in results:
    if keyboard.is_pressed('q'):
        break
    
    mensagem = False
    current_time = time.time()
    
    for box in result.boxes:
        if box.cls == 0:
            mensagem = True

    if mensagem:
        print("Pessoa detectada!")
        last_detection_time = current_time
        if not playing_sound:
            playing_sound = True
            sound_thread = threading.Thread(target=play_sound)
            sound_thread.start()
    else:
        if playing_sound:
            playing_sound = False

playing_sound = False
