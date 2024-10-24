import os
import cv2
import torch
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from weapon_label import weapon_labels
import pyrealsense2 as rs 

# Load YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
logged_threats = set()

def send_email_alert(label, confidence):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv('RECEIVER_EMAIL')
    subject = "Weapon Detected: Real-Time Alert"
    body = f"Weapon Detected: {label}\nConfidence: {confidence:.2f}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"Alert email sent for {label}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def log_weapon_detection(label, confidence):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] Weapon Detected: {label} (Confidence: {confidence:.2f})"
    with open('weapon_detection_log.txt', 'a') as log_file:
        log_file.write(log_message + '\n')

    if label not in logged_threats:
        print(log_message)
        send_email_alert(label, confidence)
        logged_threats.add(label)

def draw_boxes_on_frame(frame, detections, depth_frame=None):
    for idx in range(len(detections)):
        label = detections.iloc[idx]['name']
        confidence = detections.iloc[idx]['confidence']
        x_min = int(detections.iloc[idx]['xmin'])
        y_min = int(detections.iloc[idx]['ymin'])
        x_max = int(detections.iloc[idx]['xmax'])
        y_max = int(detections.iloc[idx]['ymax'])

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # If depth frame is provided, get the depth data
        if depth_frame is not None:
            # Calculate the depth at the center of the detected object
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            depth = depth_frame.get_distance(center_x, center_y)
            cv2.putText(frame, f'Depth: {depth:.2f}m', (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if label in weapon_labels:
            log_weapon_detection(label, confidence)

def capture_realsense_with_threat_detection(capture_delay=0.1):
    # Configure Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable both color and depth streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Run object detection on the color image
            frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            detections = results.pandas().xyxy[0]

            # Draw bounding boxes on the frame and include depth info
            draw_boxes_on_frame(color_image, detections, depth_frame)

            # Show the frames with detections
            cv2.imshow('Intel RealSense Threat Detection', color_image)

            # Introduce a delay to reduce the load
            time.sleep(capture_delay)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    capture_realsense_with_threat_detection()
