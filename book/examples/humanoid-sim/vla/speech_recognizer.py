import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr

class SpeechRecognizer(Node):
    """
    A node that listens to the microphone, transcribes speech to text, and
    publishes the result on a topic.
    """
    def __init__(self):
        super().__init__('speech_recognizer')
        self.get_logger().info('Speech Recognizer node started.')
        
        self.publisher = self.create_publisher(String, '/user_command', 10)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.get_logger().info("Calibrating for ambient noise, please wait...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.get_logger().info("Calibration complete. Ready for commands.")
        
        # Create a timer to periodically listen for commands
        self.listen_timer = self.create_timer(1.0, self.listen_for_command)

    def listen_for_command(self):
        """
        The main listening loop. This is triggered by a timer.
        """
        # We are "pausing" the timer so that it doesn't try to listen again
        # while we are already listening.
        self.listen_timer.cancel()
        
        try:
            with self.microphone as source:
                self.get_logger().info("Listening...")
                audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)
            
            self.get_logger().info("Transcribing...")
            text = self.recognizer.recognize_google(audio)
            
            self.get_logger().info(f"Recognized: '{text}'")
            
            # Publish the command
            cmd_msg = String()
            cmd_msg.data = text
            self.publisher.publish(cmd_msg)

        except sr.WaitTimeoutError:
            self.get_logger().info("No speech detected in the last 5 seconds.")
        except sr.UnknownValueError:
            self.get_logger().warn("Could not understand the audio.")
        except sr.RequestError as e:
            self.get_logger().error(f"Google Speech Recognition request failed: {e}")
        finally:
            # Restart the timer to listen for the next command
            self.listen_timer.reset()

def main(args=None):
    rclpy.init(args=args)
    speech_recognizer = SpeechRecognizer()
    rclpy.spin(speech_recognizer)
    speech_recognizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
