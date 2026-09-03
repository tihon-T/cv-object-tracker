import cv2
import numpy as np
import time

def main():

    cap = cv2.VideoCapture(0)    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    lower_color = np.array([90, 50, 50])
    upper_color = np.array([130, 255, 255])
    prev_frame_time = 0
    print("[INFO] Запуск трекера. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

      
        frame = cv2.flip(frame, 1)     
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
         
            c = max(contours, key=cv2.contourArea)
            
            
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c) 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Object Detected ({center_x}, {center_y})", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

  
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

       
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Real-Time Computer Vision Tracker", frame)

     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
