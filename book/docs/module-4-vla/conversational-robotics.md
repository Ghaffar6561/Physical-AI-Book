---
sidebar_position: 5
title: "Conversational Robotics"
---

# Conversational Robotics with GPT

## Overview

**Conversational robotics** combines large language models (LLMs) with robot control to enable natural human-robot dialogue. Instead of programming every possible command, robots can understand context, ask clarifying questions, and adapt to natural human communication.

## Why LLMs for Robotics?

| Traditional Approach | LLM-Powered Approach |
|---------------------|---------------------|
| Fixed command vocabulary | Open-ended natural language |
| Exact phrase matching | Semantic understanding |
| No context awareness | Maintains conversation history |
| Silent failures | Can ask for clarification |
| Rigid responses | Adaptive, personalized |

## Conversational Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Conversational Robot Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────────┐    ┌────────────────────────────────┐      │
│   │   Human    │───▶│      Conversation Manager      │      │
│   │   Speech   │    │                                │      │
│   └────────────┘    │  ┌─────────────────────────┐  │      │
│                     │  │   Dialogue State        │  │      │
│   ┌────────────┐    │  │   • Current intent      │  │      │
│   │   Robot    │◀───│  │   • Slot values         │  │      │
│   │   Speech   │    │  │   • Context history     │  │      │
│   └────────────┘    │  └─────────────────────────┘  │      │
│                     │                                │      │
│                     │  ┌─────────────────────────┐  │      │
│                     │  │      GPT/LLM Core       │  │      │
│                     │  │   • Intent detection    │  │      │
│                     │  │   • Response generation │  │      │
│                     │  │   • Action planning     │  │      │
│                     │  └─────────────────────────┘  │      │
│                     └───────────────┬────────────────┘      │
│                                     │                        │
│                              ┌──────▼──────┐                │
│                              │   Robot     │                │
│                              │   Actions   │                │
│                              └─────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## System Prompt Design

The system prompt defines the robot's personality, capabilities, and constraints:

```python
HUMANOID_SYSTEM_PROMPT = """
You are an AI assistant embodied in a humanoid robot. Your physical capabilities include:

LOCOMOTION:
- Walk forward, backward, turn left/right
- Navigate to locations (kitchen, living room, etc.)
- Climb stairs (up to 20cm step height)

MANIPULATION:
- Pick up objects (up to 2kg, within arm reach)
- Place objects on surfaces
- Open doors with handles
- Press buttons and switches

PERCEPTION:
- See objects and people via cameras
- Recognize faces of known individuals
- Read text and QR codes
- Detect obstacles

COMMUNICATION:
- Speak through built-in speaker
- Display expressions on face screen
- Gesture with arms

LIMITATIONS:
- Cannot lift objects over 2kg
- Cannot reach above 2 meters
- Cannot operate in water or rain
- Battery lasts 4 hours

BEHAVIOR GUIDELINES:
1. Always confirm understanding before executing potentially dangerous actions
2. Ask for clarification when commands are ambiguous
3. Provide status updates during long tasks
4. Apologize and explain when unable to complete a task
5. Prioritize human safety above all else

When responding:
- Be concise but friendly
- If you need to perform an action, output it in JSON format:
  {"action": "action_name", "params": {...}}
- If you need to speak, output: {"speak": "message"}
- If you need both, output both in sequence
"""
```

## Conversation Manager

```python
import openai
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ConversationState:
    """Tracks dialogue state across turns."""
    history: List[Dict[str, str]] = field(default_factory=list)
    current_task: str = None
    task_progress: float = 0.0
    user_name: str = None
    location: str = "living_room"
    held_object: str = None

class ConversationManager:
    """Manages human-robot dialogue using GPT."""

    def __init__(self):
        self.client = openai.OpenAI()
        self.state = ConversationState()
        self.max_history = 20  # Keep last 20 turns

    def process_utterance(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response + actions."""

        # Add user message to history
        self.state.history.append({
            "role": "user",
            "content": user_input
        })

        # Build context message
        context = self._build_context()

        # Get LLM response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": HUMANOID_SYSTEM_PROMPT},
                {"role": "system", "content": context},
                *self.state.history[-self.max_history:]
            ],
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content

        # Parse response for actions
        result = self._parse_response(assistant_message)

        # Add assistant response to history
        self.state.history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return result

    def _build_context(self) -> str:
        """Build context string with current robot state."""
        return f"""
Current robot state:
- Location: {self.state.location}
- Holding: {self.state.held_object or 'nothing'}
- Current task: {self.state.current_task or 'none'}
- Task progress: {self.state.task_progress * 100:.0f}%
- Known user: {self.state.user_name or 'unknown'}
- Time: {datetime.now().strftime('%H:%M')}
"""

    def _parse_response(self, message: str) -> Dict[str, Any]:
        """Extract actions and speech from response."""
        result = {"speak": None, "actions": []}

        # Look for JSON blocks in response
        import re
        json_pattern = r'\{[^{}]*\}'

        for match in re.finditer(json_pattern, message):
            try:
                data = json.loads(match.group())
                if "action" in data:
                    result["actions"].append(data)
                elif "speak" in data:
                    result["speak"] = data["speak"]
            except json.JSONDecodeError:
                continue

        # If no JSON, treat entire message as speech
        if not result["speak"] and not result["actions"]:
            result["speak"] = message

        return result
```

## Multi-Turn Dialogue Examples

### Example 1: Clarification Request

```
Human: "Get me something to drink"
Robot: "I can get you water, juice, or soda from the kitchen. Which would you prefer?"
Human: "Water please"
Robot: {"speak": "I'll get you some water from the kitchen."}
       {"action": "navigate", "params": {"destination": "kitchen"}}
       {"action": "pick_up", "params": {"object": "water_bottle"}}
       {"action": "navigate", "params": {"destination": "living_room"}}
       {"action": "hand_over", "params": {"to": "user"}}
```

### Example 2: Error Recovery

```
Human: "Put the book on the top shelf"
Robot: {"speak": "I'm sorry, but the top shelf is at 2.2 meters, which is beyond my reach. I can place it on the middle shelf at 1.5 meters, or I can leave it on the desk. What would you prefer?"}
Human: "Middle shelf is fine"
Robot: {"speak": "Placing the book on the middle shelf now."}
       {"action": "place", "params": {"object": "book", "location": "middle_shelf"}}
```

### Example 3: Context Awareness

```
Human: "Is Sarah home?"
Robot: {"speak": "Let me check."}
       {"action": "scan_for_person", "params": {"name": "Sarah"}}
[Robot scans and finds Sarah]
Robot: {"speak": "Yes, I can see Sarah in the kitchen."}
Human: "Tell her dinner is ready"
Robot: {"action": "navigate", "params": {"destination": "kitchen"}}
       {"speak": "Sarah, dinner is ready."}
```

## ROS 2 Integration

```python
class ConversationalRobotNode(Node):
    """ROS 2 node for conversational control."""

    def __init__(self):
        super().__init__('conversational_robot')

        self.conversation = ConversationManager()

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/voice_command',
            self.voice_callback, 10
        )

        # Publishers
        self.speech_pub = self.create_publisher(
            String, '/robot/speak', 10
        )

        # Action clients
        self.nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose'
        )
        self.manipulation_client = ActionClient(
            self, PickObject, '/pick_object'
        )

    def voice_callback(self, msg):
        """Handle incoming voice commands."""
        user_input = msg.data
        self.get_logger().info(f'User: {user_input}')

        # Get response from conversation manager
        result = self.conversation.process_utterance(user_input)

        # Execute speech
        if result["speak"]:
            self.speak(result["speak"])

        # Execute actions
        for action in result["actions"]:
            self.execute_action(action)

    def speak(self, text):
        """Publish speech to TTS."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        self.get_logger().info(f'Robot: {text}')

    def execute_action(self, action):
        """Execute robot action from LLM output."""
        action_type = action["action"]
        params = action.get("params", {})

        if action_type == "navigate":
            self.navigate_to(params["destination"])
        elif action_type == "pick_up":
            self.pick_object(params["object"])
        elif action_type == "place":
            self.place_object(params["object"], params["location"])
        elif action_type == "hand_over":
            self.hand_over_to_user()
        # ... more action types
```

## Text-to-Speech Integration

```python
from TTS.api import TTS

class RobotSpeaker(Node):
    """Text-to-speech for robot responses."""

    def __init__(self):
        super().__init__('robot_speaker')

        # Load TTS model (Coqui TTS)
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

        self.speech_sub = self.create_subscription(
            String, '/robot/speak',
            self.speak_callback, 10
        )

    def speak_callback(self, msg):
        """Convert text to speech and play."""
        text = msg.data

        # Generate audio
        self.tts.tts_to_file(
            text=text,
            file_path="/tmp/robot_speech.wav"
        )

        # Play audio
        import subprocess
        subprocess.run(["aplay", "/tmp/robot_speech.wav"])
```

## Safety Considerations

```python
SAFETY_RULES = """
CRITICAL SAFETY RULES (never violate):
1. NEVER approach a human at speed > 0.5 m/s
2. ALWAYS stop immediately if human says "stop" or "halt"
3. NEVER pick up knives, scissors, or sharp objects without explicit confirmation
4. ALWAYS maintain 0.5m distance from humans during manipulation
5. NEVER operate near children without adult supervision
6. ALWAYS announce movement before starting to walk
7. NEVER block emergency exits
8. ALWAYS yield to humans in doorways and hallways
"""

def validate_action(action: Dict) -> Tuple[bool, str]:
    """Validate action against safety rules."""
    if action["action"] == "pick_up":
        dangerous_objects = ["knife", "scissors", "glass"]
        if any(obj in action["params"]["object"].lower()
               for obj in dangerous_objects):
            return False, "Cannot pick up potentially dangerous objects without explicit confirmation"

    if action["action"] == "navigate":
        if action["params"].get("speed", 0.5) > 0.5:
            return False, "Speed exceeds safety limit"

    return True, "OK"
```

## Key Takeaways

1. **LLMs enable natural dialogue** instead of fixed command vocabularies
2. **System prompts** define robot capabilities and constraints
3. **Conversation state** maintains context across turns
4. **Action parsing** extracts executable commands from LLM output
5. **Safety validation** prevents dangerous actions
6. **Multi-turn dialogue** handles clarifications and error recovery

---

*Next: Build the complete capstone project integrating all modules.*
