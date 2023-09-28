import boto3
import io
import paho.mqtt.client as mqtt
import yaml
import json
import requests
import time
import os
from datetime import datetime
from PIL import Image

# Define the log file paths
log_file_path = "file.log"
facialrecognition_log_file_path = "facialrecognition.log"

# Create or overwrite the facialrecognition.log file at startup
with open(facialrecognition_log_file_path, "w") as fr_log_file:
    fr_log_file.write("Facial Recognition Log initialized\n")

last_api_call_time = 0

rekognition = boto3.client('rekognition', region_name='eu-west-2')
dynamodb = boto3.client('dynamodb', region_name='eu-west-2')

def process_image(image_path):
    time.sleep(5)  # Delay before sending the image to AWS for recognition
    print("Processing image:", image_path)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    image = Image.open(image_path)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    image_binary = stream.getvalue()
    
    try:
        print("Sending image for face detection...")
        response = rekognition.detect_faces(Image={'Bytes': image_binary})
        if 'FaceDetails' in response:
            print(f"Detected {len(response['FaceDetails'])} faces in the image.")
            all_faces = response['FaceDetails']
            image_width = image.size[0]
            image_height = image.size[1]
            for face in all_faces:
                box = face['BoundingBox']
                x1 = int(box['Left'] * image_width) * 0.9
                y1 = int(box['Top'] * image_height) * 0.9
                x2 = int(box['Left'] * image_width + box['Width'] * image_width) * 1.10
                y2 = int(box['Top'] * image_height + box['Height'] * image_height) * 1.10
                image_crop = image.crop((x1, y1, x2, y2))
                stream_crop = io.BytesIO()
                image_crop.save(stream_crop, format="JPEG")
                image_crop_binary = stream_crop.getvalue()
                
                print("Searching for matches in the face collection...")
                response_search = rekognition.search_faces_by_image(CollectionId='home_access', Image={'Bytes': image_crop_binary})
                with open(facialrecognition_log_file_path, "a") as fr_log_file:
                    if len(response_search['FaceMatches']) > 0:
                        for match in response_search['FaceMatches']:
                            face_data = dynamodb.get_item(TableName='facerecognition', Key={'RekognitionId': {'S': match['Face']['FaceId']}})
                            if 'Item' in face_data:
                                person = face_data['Item']['FullName']['S']
                                fr_log_file.write(f"{current_time} - Recognized: {person}, Confidence: {match['Face']['Confidence']}\n")
                            else:
                                fr_log_file.write(f"{current_time} - No match found for a face\n")
                    else:
                        fr_log_file.write(f"{current_time} - Face at coordinates {box} cannot be recognized\n")
        else:
            print("No faces detected in the image.")
    except Exception as e:
        print("Error during the image processing:", str(e))
    
    # Delete old images
    image_files = sorted([f for f in os.listdir() if f.endswith('.jpg')], key=os.path.getctime)
    while len(image_files) > 50:
        os.remove(image_files[0])
        del image_files[0]

def generate_and_download_snapshot(api_key, serial):
    global last_api_call_time
    current_time = time.time()
    if current_time - last_api_call_time < 10:
        print("Waiting for 10 seconds before making another API call.")
        return None
    
    time.sleep(5)  # 5-second delay after MQTT confidence trigger and before the actual API call
    
    url = f"https://api.meraki.com/api/v1/devices/{serial}/camera/generateSnapshot"
    headers = {"X-Cisco-Meraki-API-Key": api_key}
    response = requests.post(url, headers=headers)
    last_api_call_time = current_time
    
    if response.status_code == 202:
        max_retries = 10
        retries = 0
        snapshot_url = None
        
        while retries < max_retries:
            snapshot_info = response.json()
            snapshot_url = snapshot_info.get("url")
            if snapshot_url:
                time.sleep(5)
                imageName = f"{datetime.now().strftime('%d%m%Y_%H%M%S')}-{serial}.jpg"
                image_response = requests.get(snapshot_url)
                if image_response.status_code == 200:
                    with open(imageName, 'wb') as file:
                        file.write(image_response.content)
                    print(f"Image downloaded and saved as {imageName}")
                    return imageName
                else:
                    print(f"Failed to download image. Status Code: {image_response.status_code}")
                    return None
            else:
                print("Snapshot URL not found in response content. Retrying in 5 seconds...")
                time.sleep(5)
                retries += 1
        
        if retries >= max_retries:
            print("Maximum retry count reached. Exiting...")
            return None
    else:
        print(f"Failed to generate snapshot. Status Code: {response.status_code}. Response: {response.text}")
        return None

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("#")

def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())
        if "objects" in data and len(data["objects"]) == 1:
            person = data["objects"][0]
            if person["type"] == "person" and person["confidence"] > 70:
                serial = message.topic.split("/")[-2]
                api_key = 'KEY HERE'
                imageName = generate_and_download_snapshot(api_key, serial)
                if imageName:
                    process_image(imageName)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Topic: {message.topic}, Message: {message.payload.decode()}\n")

with open("broker_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(config["mqtt_broker_host"], config["mqtt_broker_port"], 60)
mqtt_client.loop_forever()
