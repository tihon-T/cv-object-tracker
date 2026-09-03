import cv2
import numpy as np
import time

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error. Camera is not found")
        return

    # Окно с ползунками для точной настройки цвета под твою комнату
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 90, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 130, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    prev_frame_time = 0

    print("[INFO] Настрой ползунки под свой предмет. Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Размытие кадра, чтобы убрать мелкий шум и тряску
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Считываем значения ползунков
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_color = np.array([l_h, l_s, l_v])
        upper_color = np.array([u_h, u_s, u_v])

        # Создаем маску
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Очистка маски от точечного мусора
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            
            # Подняли порог площади до 1500, чтобы мелкие объекты на фоне вообще не учитывались
            if cv2.contourArea(c) > 1500:
                x, y, w, h = cv2.boundingRect(c)
                
                # Рисуем четкую рамку
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                cv2.putText(frame, f"Target Locked [{center_x}, {center_y}]", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Object Tracker", frame)
        cv2.imshow("Mask (Debug)", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
