import os
import cv2
import torch
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from onvif import ONVIFCamera
from weapon_label import weapon_labels

# Load YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
logged_threats = set()


def get_rtsp_uri(camera_ip, camera_port, username, password):
    try:
        mycam = ONVIFCamera(camera_ip, camera_port, username, password)
        media_service = mycam.create_media_service()
        profiles = media_service.GetProfiles()

        stream_uri = media_service.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'},
            },
            'ProfileToken': profiles[0].token
        })
        return stream_uri['Uri']
    except Exception as e:
        print(f'Error obtaining RTSP URI: {e}')
        return None


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


def draw_boxes_on_frame(frame, detections):
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

        if label in weapon_labels:
            log_weapon_detection(label, confidence)


def capture_rtsp_stream_with_threat_detection(rtsp_url, capture_delay=10):
    capture = cv2.VideoCapture(rtsp_url)

    if not capture.isOpened():
        print(f"Error opening video stream {rtsp_url}")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print('Failed to grab frame')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.pandas().xyxy[0]
        draw_boxes_on_frame(frame, detections)

        cv2.imshow('Aiges A Physical Threat Detection System', frame)

        # Introduce a delay to reduce the load
        time.sleep(capture_delay)

        # Check if 'q' is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


# ------------------- Main Execution -------------------
if __name__ == "__main__":
    # Camera credentials
    # camera_ip = '192.168.1.10'
    # camera_port = 80
    # username = 'admin'
    # password = 'password'

    rtsp_url = 'rtsp://rtspstream:7ef241fbeb47fa94786eaeee49523e8b@zephyr.rtsp.stream/movie'
    capture_rtsp_stream_with_threat_detection(rtsp_url)
