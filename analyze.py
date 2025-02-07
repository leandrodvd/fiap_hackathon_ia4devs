import cv2
from ultralytics import YOLO
import requests
import argparse

def send_alert(mail_gun_api_key, mail_gun_domain, target_email):
    response = requests.post(
        f"https://api.mailgun.net/v3/{mail_gun_domain}/messages",
        auth=("api", mail_gun_api_key),
        data={
            "from": "Alerta de segurança <hackathon@example.com>",
            "to": target_email,
            "subject": "🚨 Alerta de segurança: Arma detectada!",
            "text": f"Uma arma foi detectada pelo sistema de video de segurança.",
        },
    )

    if response.status_code == 200:
        print(f"📧 Alerta enviado enviado:Arma detectada! - verifique o email em: https://maildrop.cc/inbox/?mailbox=leandro_hackathon_fiap_ia4devs")
    else:
        print(f"❌ Falha ao enviar email de alerta: {response.text}")


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Detect weapons in a video and send alerts.")
parser.add_argument("video_path", type=str, help="Path to the input video file.")
parser.add_argument("mail_gun_api_key", type=str, help="api key for mail gun service")
parser.add_argument("mail_gun_domain", type=str, help="domain for mail gun service")
parser.add_argument("notification_email", type=str, help="demaili to receive notifications")
args = parser.parse_args()

mail_gun_api_key = args.mail_gun_api_key
mail_gun_domain = args.mail_gun_domain
notification_email = args.notification_email

send_alert(mail_gun_api_key, mail_gun_domain, notification_email)
# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Adjust path if needed

# Open the video file
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video (optional: save with bounding boxes)
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

weapon_detected = False
# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Run YOLOv8 detection
    results = model(frame)
    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            confidence = result.boxes.conf[0].item()  # Get confidence score
            if confidence > 0.2:
                weapon_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Detection", frame)

    # Save the frame to the output video
    out.write(frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

if(weapon_detected):
    send_alert()