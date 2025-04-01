import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands com modelo mais pesado para maior precisão
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Função para verificar se um dedo está levantado
def is_finger_up(hand_landmarks, finger_tip, finger_dip):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_dip].y

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

while True:
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # Converter a imagem para RGB (necessário para o MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Processar resultados
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = results.multi_handedness[i].classification[0].label  # "Left" ou "Right"
            
            # Identificar se os dedos estão levantados
            fingers = []
            if hand_label == "Right":
                fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)  # Polegar mão direita
            else:
                fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)  # Polegar mão esquerda
            
            fingers.append(is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP))
            fingers.append(is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP))
            fingers.append(is_finger_up(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP))
            fingers.append(is_finger_up(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP))
            
            # Exibir o status dos dedos na imagem
            for i, is_up in enumerate(fingers):
                status = "Up" if is_up else "Down"
                cv2.putText(image, f"{hand_label} Hand - Finger {i+1}: {status}", (10, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Mao', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()