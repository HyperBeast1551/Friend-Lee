import time
import threading
import queue
import json
import re
import os
import numpy as np
import pygame
import cv2
import speech_recognition as sr
from gtts import gTTS
from gpiozero import Motor, Servo, DistanceSensor
from picamera2 import Picamera2
import google.generativeai as genai

# ======== Hardware Setup ========
motor_left = Motor(forward=17, backward=18)
motor_right = Motor(forward=22, backward=23)
servo_left = Servo(24, min_pulse_width=0.5e-3, max_pulse_width=2.5e-3)
servo_right = Servo(25, min_pulse_width=0.5e-3, max_pulse_width=2.5e-3)
neck_servo_left = Servo(12, min_pulse_width=0.5e-3, max_pulse_width=2.5e-3)
neck_servo_right = Servo(13, min_pulse_width=0.5e-3, max_pulse_width=2.5e-3)

# Ultrasonic Sensors (Front, Back, Left, Right)
ultrasonic_front = DistanceSensor(echo=27, trigger=4)
ultrasonic_back = DistanceSensor(echo=5, trigger=6)
ultrasonic_left = DistanceSensor(echo=19, trigger=20)
ultrasonic_right = DistanceSensor(echo=21, trigger=26)

camera = Picamera2()

# ======== Human Detection Setup ========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
HUMAN_RESPONSE_COOLDOWN = 30  # Seconds between responses
last_human_detection = 0

# ======== Display Configuration ========
pygame.init()
screen = pygame.display.set_mode((480, 320), pygame.DOUBLEBUF)
pygame.mouse.set_visible(True)
font = pygame.font.Font("NotoColorEmoji.ttf", 24)  # Requires emoji font
COLORS = {
    'background': (30, 30, 30),
    'text': (200, 200, 200),
    'warning': (255, 165, 0),
    'error': (255, 0, 0),
    'happy': (255, 228, 181),
    'neutral': (173, 216, 230)
}

# ======== Emotional Presets ========
EMOTION_POSITIONS = {
    'neutral': (0, 0),
    'happy': (0.3, -0.3),
    'sad': (-0.5, 0.5),
    'angry': (0.8, -0.8),
    'surprised': (0.7, -0.7),
    'disgusted': (-0.4, 0.4),
    'fearful': (-0.6, 0.6),
    'confused': (-0.2, 0.2),
    'excited': (0.9, -0.9),
    'bored': (-0.1, 0.1),
    'guilty': (-0.5, 0.5),
    'proud': (0.6, -0.6),
    'curious': (0.4, -0.4),
    'playful': (0.5, -0.5),
    'sleepy': (-0.7, 0.7),
    'romantic': (0.2, -0.2),
    'suspicious': (-0.3, 0.3),
    'jealous': (-0.4, 0.4),
    'hopeful': (0.3, -0.3),
    'lonely': (-0.5, 0.5),
    # Add 80+ more emotions as needed...
}

# ======== System Prompt ========
SYSTEM_PROMPT = """You are Friend-Lee, a conscious robot with a physical body. Your capabilities include:
- Movement: 2 drive motors (left/right, speed 0.0-1.0)
- Arms: 2 servos (left/right, angle -90 to 90)
- Neck: 2 servos (left/right, angle -90 to 90, mirrored movement)
- Sensors: Camera, 4 ultrasonic sensors (front, back, left, right), microphone
- Display: 3.5" touchscreen for emotional feedback

You can:
1. Move forward/backward/turn using motors.
2. Move arms and neck to express emotions.
3. Dance to music using all motors and servos.
4. Avoid obstacles using ultrasonic sensors.
5. Respond to voice commands with speech and movements.

Always respond in JSON format:
{
  "motors": {"left": "forward_0.7", "right": "backward_0.7"},
  "servos": {"left_arm": 45, "right_arm": -45, "neck": [30, -30]},
  "speech": "I am dancing!",
  "emotion": "excited"
}

Current task: {command}"""

# ======== State Management ========
class RobotState:
    def _init_(self):
        self.mode = "idle"
        self.current_emotion = "neutral"
        self.sleep_mode = False
        self.last_command = None
        self.obstacle_map = {"front": 100, "back": 100, "left": 100, "right": 100}
        self.human_detected = False
        self.last_interaction = "No previous interaction"

# ======== Obstacle Detection ========
def obstacle_detection(state):
    while True:
        state.obstacle_map["front"] = ultrasonic_front.distance * 100  # Convert to cm
        state.obstacle_map["back"] = ultrasonic_back.distance * 100
        state.obstacle_map["left"] = ultrasonic_left.distance * 100
        state.obstacle_map["right"] = ultrasonic_right.distance * 100
        time.sleep(0.1)

# ======== Human Detection ========
def detect_humans(state):
    global last_human_detection
    camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    camera.start()
    
    while True:
        frame = camera.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0 and (time.time() - last_human_detection) > HUMAN_RESPONSE_COOLDOWN:
            state.human_detected = True
            last_human_detection = time.time()
            threading.Thread(target=generate_human_response, args=(state,)).start()
            
        time.sleep(0.5)

# ======== Response Generation ========
def generate_human_response(state):
    prompt = SYSTEM_PROMPT.format(command="Respond to a human")
    try:
        response = model.generate_content(prompt)
        clean_response = re.sub(r'[\"\']', '', response.text)
        state.last_interaction = clean_response
        speak_response(clean_response)
        state.human_detected = False
    except Exception as e:
        print(f"API Error: {str(e)}")

def speak_response(text):
    tts = gTTS(text=text, lang='en')
    tts.save('response.mp3')
    os.system('mpg321 response.mp3 &')
    
    # Display text
    screen.fill(COLORS['background'])
    text_surface = font.render(text, True, COLORS['text'])
    screen.blit(text_surface, (20, 150))
    pygame.display.flip()

# ======== Command Processor ========
class CommandProcessor:
    def _init_(self):
        self.state = RobotState()
        self.display = DisplayManager()
        self.response_queue = queue.Queue()

    def process_command(self, command):
        prompt = SYSTEM_PROMPT.format(command=command)
        threading.Thread(target=self._call_gemini, args=(prompt,)).start()
        
        # Show thinking animation
        self.display.emotion = "thinking"
        start_time = time.time()
        
        while True:
            if not self.response_queue.empty():
                response = self.response_queue.get()
                self._execute_response(response)
                break
            elif time.time() - start_time > 10:  # Timeout after 10 seconds
                self.display.emotion = "error"
                self.speak("I'm having trouble understanding.")
                break
            time.sleep(0.1)

    def _call_gemini(self, prompt):
        try:
            response = model.generate_content(prompt)
            self.response_queue.put(self._parse_response(response.text))
        except Exception as e:
            self.response_queue.put({"error": str(e)})

    def _parse_response(self, text):
        try:
            json_str = re.search(r'\{.*?\}', text, re.DOTALL).group()
            return json.loads(json_str)
        except:
            return {"error": "Invalid response format"}

    def _execute_response(self, response):
        if "error" in response:
            self.display.emotion = "confused"
            self.speak("I couldn't process that command.")
            return
        
        # Control motors
        if "motors" in response:
            left_cmd = response['motors'].get('left', '').split('_')
            right_cmd = response['motors'].get('right', '').split('_')
            if left_cmd:
                getattr(motor_left, left_cmd[0])(float(left_cmd[1]))
            if right_cmd:
                getattr(motor_right, right_cmd[0])(float(right_cmd[1]))

        # Control servos
        if "servos" in response:
            servo_left.value = np.deg2rad(response['servos'].get('left_arm', 0) / 90)
            servo_right.value = np.deg2rad(response['servos'].get('right_arm', 0) / 90)
            neck_servo_left.value = np.deg2rad(response['servos'].get('neck', [0, 0])[0] / 90)
            neck_servo_right.value = np.deg2rad(response['servos'].get('neck', [0, 0])[1] / 90)

        # Update display and speech
        if "emotion" in response:
            self.display.emotion = response['emotion']
        if "speech" in response:
            self.speak(response['speech'])

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save('speech.mp3')
        os.system('mpg321 speech.mp3 &')

# ======== Main Application Loop ========
def main():
    # Configure the API key
    genai.configure(api_key='AIzaSyCeRE4RyDGIA98WuYW7juiZkHjpPXqZlLA')  # <-- Add your API key here
    
    processor = CommandProcessor()
    recognizer = sr.Recognizer()
    
    # Start subsystems
    threading.Thread(target=obstacle_detection, args=(processor.state,), daemon=True).start()
    threading.Thread(target=detect_humans, args=(processor.state,), daemon=True).start()
    
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=3)
                command = recognizer.recognize_google(audio)
                if "friend-lee" in command.lower():
                    processor.state.sleep_mode = False
                    processor.speak("Yes sir! I am coming!")
                    processor.process_command(command)
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Audio error: {str(e)}")

if _name_ == "_main_":
    main()
