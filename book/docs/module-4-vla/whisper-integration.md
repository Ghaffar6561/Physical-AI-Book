---
sidebar_position: 4
title: "Whisper Voice-to-Action"
---

# Voice-to-Action with OpenAI Whisper

## Overview

**Voice-to-Action** enables humanoid robots to receive natural language voice commands and translate them into physical actions. OpenAI's Whisper model provides state-of-the-art speech recognition that works across languages, accents, and noisy environments—essential for robots operating in real-world settings.

## Why Whisper for Robotics?

| Feature | Benefit for Humanoids |
|---------|----------------------|
| **Multilingual** | Operate globally without retraining |
| **Noise Robust** | Works in factories, homes, public spaces |
| **Open Source** | Deploy on-device (Jetson) |
| **Low Latency** | Real-time command processing |
| **Accurate** | Near-human transcription quality |

## Voice-to-Action Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Voice-to-Action Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│   │ Microphone │───▶│  Whisper   │───▶│  Command   │        │
│   │   Array    │    │    ASR     │    │   Parser   │        │
│   └────────────┘    └────────────┘    └─────┬──────┘        │
│                                             │                │
│                                      ┌──────▼──────┐        │
│                                      │     LLM     │        │
│                                      │  (GPT/Local)│        │
│                                      └──────┬──────┘        │
│                                             │                │
│   ┌────────────┐    ┌────────────┐    ┌─────▼──────┐        │
│   │   Robot    │◀───│   Action   │◀───│   Task     │        │
│   │ Controller │    │  Executor  │    │  Planner   │        │
│   └────────────┘    └────────────┘    └────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installing Whisper

### On Workstation (GPU)

```bash
pip install openai-whisper

# Or for faster inference
pip install faster-whisper
```

### On Jetson (Edge Deployment)

```bash
# Install with CUDA support
pip install openai-whisper

# Or use whisper.cpp for C++ deployment
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make WHISPER_CUDA=1

# Download models
bash ./models/download-ggml-model.sh base.en
```

## ROS 2 Whisper Node

```python
# whisper_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_interfaces.msg import AudioData
import whisper
import numpy as np
import threading
import queue

class WhisperNode(Node):
    """ROS 2 node for speech-to-text using Whisper."""

    def __init__(self):
        super().__init__('whisper_node')

        # Parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'en')
        self.declare_parameter('device', 'cuda')

        model_size = self.get_parameter('model_size').value
        device = self.get_parameter('device').value

        # Load Whisper model
        self.get_logger().info(f'Loading Whisper {model_size} model...')
        self.model = whisper.load_model(model_size, device=device)
        self.get_logger().info('Whisper model loaded')

        # Audio buffer
        self.audio_buffer = queue.Queue()
        self.sample_rate = 16000

        # Subscribers and publishers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/microphone/audio',
            self.audio_callback,
            10
        )

        self.text_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def audio_callback(self, msg):
        """Receive audio chunks from microphone."""
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        self.audio_buffer.put(audio_float)

    def process_audio(self):
        """Process audio buffer with Whisper."""
        accumulated_audio = []
        silence_threshold = 0.5  # seconds of silence to trigger transcription

        while rclpy.ok():
            try:
                chunk = self.audio_buffer.get(timeout=0.1)
                accumulated_audio.append(chunk)

                # Check for voice activity
                if self.detect_silence(chunk) and len(accumulated_audio) > 10:
                    # End of utterance detected
                    audio_array = np.concatenate(accumulated_audio)

                    # Transcribe
                    result = self.transcribe(audio_array)

                    if result and len(result) > 0:
                        msg = String()
                        msg.data = result
                        self.text_pub.publish(msg)
                        self.get_logger().info(f'Transcribed: "{result}"')

                    accumulated_audio = []

            except queue.Empty:
                continue

    def transcribe(self, audio):
        """Run Whisper transcription."""
        result = self.model.transcribe(
            audio,
            language=self.get_parameter('language').value,
            task='transcribe',
            fp16=True  # Use FP16 on GPU
        )
        return result['text'].strip()

    def detect_silence(self, audio, threshold=0.01):
        """Detect if audio chunk is silence."""
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold


def main():
    rclpy.init()
    node = WhisperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Streaming Whisper for Low Latency

```python
from faster_whisper import WhisperModel

class StreamingWhisperNode(Node):
    """Low-latency streaming Whisper transcription."""

    def __init__(self):
        super().__init__('streaming_whisper')

        # Use faster-whisper for streaming
        self.model = WhisperModel(
            "base",
            device="cuda",
            compute_type="float16"
        )

        self.audio_buffer = []
        self.buffer_duration = 0.5  # Process every 0.5 seconds

    def transcribe_stream(self, audio_chunk):
        """Streaming transcription with partial results."""
        self.audio_buffer.append(audio_chunk)

        # Accumulated enough audio
        if len(self.audio_buffer) * 0.1 >= self.buffer_duration:
            audio = np.concatenate(self.audio_buffer)

            segments, info = self.model.transcribe(
                audio,
                beam_size=1,  # Faster
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            for segment in segments:
                yield segment.text

            # Keep last chunk for context
            self.audio_buffer = self.audio_buffer[-1:]
```

## Wake Word Detection

Before running Whisper continuously, use a wake word:

```python
import pvporcupine

class WakeWordDetector:
    """Detect wake word before activating Whisper."""

    def __init__(self, wake_word="humanoid"):
        self.porcupine = pvporcupine.create(
            access_key='YOUR_ACCESS_KEY',
            keywords=[wake_word]
        )
        self.is_listening = False

    def process_audio(self, audio_frame):
        """Check for wake word in audio frame."""
        keyword_index = self.porcupine.process(audio_frame)

        if keyword_index >= 0:
            self.is_listening = True
            return True

        return False
```

## Complete Voice-to-Action Pipeline

```python
class VoiceToActionController(Node):
    """Complete pipeline from voice to robot action."""

    def __init__(self):
        super().__init__('voice_to_action')

        # Components
        self.wake_detector = WakeWordDetector()
        self.whisper = WhisperModel("base", device="cuda")
        self.llm_client = openai.OpenAI()

        # Action client
        self.action_client = ActionClient(
            self, ExecuteTask, '/humanoid/execute_task'
        )

        # Audio subscription
        self.audio_sub = self.create_subscription(
            AudioData, '/microphone/audio',
            self.audio_callback, 10
        )

        self.listening = False
        self.audio_buffer = []

    def audio_callback(self, msg):
        audio = np.frombuffer(msg.data, dtype=np.int16)

        if not self.listening:
            # Check for wake word
            if self.wake_detector.process_audio(audio):
                self.listening = True
                self.audio_buffer = []
                self.speak("Yes?")  # Acknowledge

        else:
            # Accumulate audio for transcription
            self.audio_buffer.append(audio)

            if self.detect_end_of_speech():
                self.process_command()

    def process_command(self):
        """Transcribe and execute voice command."""
        # Combine audio
        audio = np.concatenate(self.audio_buffer).astype(np.float32) / 32768.0

        # Transcribe
        segments, _ = self.whisper.transcribe(audio)
        command = " ".join([s.text for s in segments]).strip()

        self.get_logger().info(f'Command: "{command}"')

        # Convert to action plan using LLM
        plan = self.command_to_plan(command)

        # Execute
        for action in plan:
            self.execute_action(action)

        self.listening = False

    def command_to_plan(self, command):
        """Use LLM to convert command to action sequence."""
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ROBOT_PLANNER_PROMPT},
                {"role": "user", "content": command}
            ],
            response_format={"type": "json_object"}
        )

        import json
        return json.loads(response.choices[0].message.content)['actions']

    def speak(self, text):
        """Text-to-speech feedback."""
        # Publish to TTS node
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
```

## Model Size Selection

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| `tiny` | 39M | 1GB | 32x | Good |
| `base` | 74M | 1GB | 16x | Better |
| `small` | 244M | 2GB | 6x | High |
| `medium` | 769M | 5GB | 2x | Very High |
| `large` | 1550M | 10GB | 1x | Best |

**Recommendation for Humanoids:**
- **Jetson Orin Nano**: `tiny` or `base`
- **Jetson Orin NX/AGX**: `small` or `medium`
- **Workstation**: `medium` or `large`

## Key Takeaways

1. **Whisper** provides robust speech-to-text for noisy robot environments
2. **Wake word detection** prevents continuous processing
3. **Streaming** reduces latency for real-time interaction
4. **Edge deployment** is possible on Jetson with smaller models
5. **LLM integration** converts natural language to action plans

---

*Next: Learn GPT integration for conversational robotics.*
