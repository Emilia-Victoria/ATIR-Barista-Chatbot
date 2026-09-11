#!/usr/bin/env python3
"""
Virtual Robot Face with MAR Detection and FAST Speech Recognition
Ultra-fast streaming speech recognition with real-time transcription feedback.
"""
import asyncio
import time
import sys
from robot_face_system import RobotFace, RobotState
from face_detector import FaceDetector
from speech_handler import SpeechDetector
from voice_handler import RobotVoice
from llm_handler import AsyncLLMHandler, LLMResponse
import threading

class Robot:
    """Manages robot state and responses"""
    robot_face = None
    face_detection = None
    speech_detection = None
    robot_voice = None
    llm_handler = None

    # State tracking
    last_speaking = False
    last_face_visible = False
    conversation_started = False
    mouth_speaking = None
    face_visible = None
    head_x = None
    head_y = None
    mar_value = None

    def __init__(self):
        """Initialize all systems."""
        print("Initializing robot")
        self.robot_face = RobotFace()
        self.robot_face.start_animation("idle")
        self.face_detection = FaceDetector(speaking_threshold=0.04)
        self.speech_detection = SpeechDetector()
      
        self.robot_voice = RobotVoice(
            on_speech_start=self.handleRobotVoiceStarted,
            on_speech_end=self.handleRobotVoiceEnded,
            on_speech_interrupted=self.handleRobotVoiceInterrupted,
        )
    
        self.llm_handler = AsyncLLMHandler()
        self.face_detection.start()

        speech_thread = threading.Thread(target=self.start_speech_detection, daemon=True)
        speech_thread.start()
        print("Started background speech detection")

        # Let systems initialize
        time.sleep(1.0) 


    def updateFaceRecognitionInfo(self):
        # Get tracking data
        status = self.face_detection.get_status()
        self.mouth_speaking = status['speaking']
        self.face_visible = status['face_visible']
        self.head_x = status['head_x']
        self.head_y = status['head_y']
        self.mar_value = status['mar']
 
        # Handle face detection changes
        if self.face_visible != self.last_face_visible:
            if self.face_visible:
                print("👤 FACE DETECTED")

                # Start conversation when a face appears
                if not self.conversation_started:
                    self.conversation_started = True
                    time.sleep(1.0) 
                    self.makeTheRobotSay(
                    "Welcome to Café Analog! How can I help you?"
                    )
            else:
                print("👤 FACE LOST")
            self.last_face_visible = self.face_visible


    async def handle_human_speech_end(self, transcription: str):
        print(f"END: '{transcription}'")
        if transcription:
            self.processWithLLM(transcription)

    def start_speech_detection(self):
        """this method starts and runs the async voice activity detection in a separate thread"""  
        async def speech_loop():
            speech_handler = self.speech_detection

            if not speech_handler.start():
                print("Failed to start speech handler")
                return
            
            try:
                await speech_handler.listen_with_voice_activity_detection(
                    voice_threshold=-40,
                    silence_timeout=3.0,
                    on_end=self.handle_human_speech_end
                )
            finally:
                speech_handler.stop()
        
        asyncio.run(speech_loop())


    def handle_LLM_chunk(self, chunk: str):
        print(chunk, end='', flush=True)

    def handle_LLM_complete(self, response: LLMResponse):
        print(f"\nComplete response received: {response.content}")
        if response.success and response.content:
            self.makeTheRobotSay(response.content)

    def handleRobotVoiceEnded(self):
        print("\n unpausing speech recognition \n")
        self.speech_detection.pause_voice_detection(False)

    def handleRobotVoiceStarted(self, text):
        # Pause transcription so the robot does not transcribe itself.
        print("\n pausing speech recognition \n")
        self.speech_detection.pause_voice_detection(True)

    def handleRobotVoiceInterrupted(self):
        print("\n unpausing speech recognition after interruption \n")
        self.speech_detection.pause_voice_detection(False)

    def processWithLLM(self, utterance):
        asyncio.get_event_loop().create_task(self.llm_handler.send_prompt_streaming(
            utterance, 
            callback=self.handle_LLM_chunk,
            final_callback=self.handle_LLM_complete
        ))

    def makeTheRobotSay(self, utterance):
        if not utterance:
            return
        # Close the microphone gate before audio playback can begin. The voice
        # callbacks keep it closed for the complete queued speech burst.
        self.speech_detection.pause_voice_detection(True)
        self.robot_voice.speak(utterance)

    def reactWithRobotFace(self):
        self.robot_face.loop()
        
        if self.robot_voice.is_speaking():
            self.robot_face.set_state(RobotState.SPEAKING)
        else:
            self.robot_face.set_state(RobotState.IDLE)

    def run_loop(self):
        """Main loop"""
        try:
            while self.robot_face.handle_events():
                self.updateFaceRecognitionInfo()
                self.reactWithRobotFace()
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        self.face_detection.stop()
        self.speech_detection.stop()
        self.robot_voice.shutdown()
        self.robot_face.shutdown()
        


def main():
    """Main entry point."""
    robot = Robot()
    robot.run_loop()
 

if __name__ == "__main__":
    sys.exit(main())
