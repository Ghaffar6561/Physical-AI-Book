# Setting Up Local Language Models with Ollama

This guide covers installing and running open-source language models locally. This is essential for VLA development when you can't use API services (cost, privacy, offline), or want to iterate quickly.

---

## Why Local LLMs?

| Aspect | API (GPT-4) | Local (Ollama + Llama 2) |
|--------|-----------|------------------------|
| **Cost** | $0.03 per request | Free (once downloaded) |
| **Latency** | 1-3 seconds | 100-500ms (depends on GPU) |
| **Privacy** | Data sent to OpenAI | Stays on your machine |
| **Offline** | Requires internet | Works offline |
| **Accuracy** | Best-in-class | Good (70-85% of GPT-4) |
| **Setup** | 5 minutes | 30 minutes |

**Best for robotics**: Local models during development/iteration. GPT-4 for final validation.

---

## Installation

### Option 1: Ollama (Recommended, Easiest)

**What is Ollama?** Simple CLI tool to download and run LLMs locally.

**Step 1: Download Ollama**

```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

**Step 2: Verify Installation**

```bash
ollama --version
# Output: ollama version 0.1.0
```

**Step 3: Download a Model**

```bash
# Download Llama 2 7B (4GB, ~30 seconds on fast internet)
ollama pull llama2

# Or Mistral (5B, faster)
ollama pull mistral

# Or Llama 2 13B (for better quality, 8GB)
ollama pull llama2:13b
```

**Step 4: Run the Model**

```bash
# Start the server (runs on localhost:11434)
ollama serve

# In another terminal, test it
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why are robots cool?",
  "stream": false
}'
```

**Output**:
```json
{
  "response": "Robots are cool because they can perform repetitive tasks with precision, improve safety, and extend human capabilities...",
  "total_duration": 2345678000
}
```

### Option 2: LM Studio (GUI Option)

If you prefer a graphical interface:

```bash
# Download from https://lmstudio.ai/
# 1. Install LM Studio
# 2. Open app
# 3. Search for "llama-2" or "mistral"
# 4. Click Download
# 5. Click "Load Model"
# 6. Go to Local Server → Start Server
# 7. API available at http://localhost:1234
```

### Option 3: Hugging Face Transformers (For Developers)

If you want fine-grained control:

```bash
pip install transformers torch

# Download and cache model (first time: ~10GB download, takes 5-10 minutes)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

# Inference
inputs = tokenizer('Hello, how are', return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
"
```

---

## Choosing the Right Model

```
Model         | Size | Speed | Quality | Memory | Best For
──────────────|------|-------|---------|--------|──────────
Phi 2.7B      | 2GB  | Fast  | Fair    | 3GB    | Edge devices, phone
Mistral 7B    | 4GB  | Fast  | Good    | 6GB    | Development
Llama 2 7B    | 4GB  | Fast  | Good    | 6GB    | Standard choice
Llama 2 13B   | 8GB  | Med   | Better  | 10GB   | Quality > speed
Llama 2 70B   | 40GB | Slow  | Great   | 42GB   | Best output (needs good GPU)
```

**For VLA robotics development**: **Mistral 7B** or **Llama 2 7B**

---

## Using Local LLMs in Your Code

### Python API with Ollama

```python
import requests
import json

def call_llama(prompt: str, model: str = "llama2") -> str:
    """Call local Llama 2 via Ollama."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,  # Creativity (0-1)
            "top_p": 0.9,        # Diversity
        }
    )

    return response.json()["response"]

# Usage
result = call_llama("Analyze this robot scene: [describe scene]")
print(result)
```

### OpenAI-Compatible API

```python
from openai import OpenAI

# Point to local Ollama server
client = OpenAI(
    api_key="ollama",  # Dummy key
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": "What should the robot do?"},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Vision-Language Model (LLaVA)

For image understanding locally:

```bash
# Pull LLaVA (vision-language model)
ollama pull llava

# This lets you do vision + language tasks locally
```

**Usage**:
```python
import ollama

# Load image as base64
import base64
with open("robot_scene.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = ollama.generate(
    model="llava",
    prompt="Analyze this robot scene and suggest an action",
    images=[image_base64],
    stream=False
)

print(response["response"])
```

---

## Performance Tuning

### GPU Acceleration

By default, Ollama uses CPU. To use GPU:

```bash
# NVIDIA GPU
# Install CUDA drivers first, then:
ollama pull llama2  # Will auto-detect GPU

# AMD GPU
# Install ROCm, then:
ROCM_HOME=/opt/rocm ollama serve

# Apple Silicon (M1/M2/M3)
# Automatic GPU acceleration (no extra setup needed)
```

**Performance differences**:
```
CPU only:    1-2 tokens/second (slow)
GPU (RTX 3060):  10-20 tokens/second (good)
GPU (RTX 3090):  30-50 tokens/second (fast)
```

### Memory Management

```bash
# Keep model in memory (default)
ollama serve

# Or unload after request (saves RAM)
# Edit /etc/systemd/system/ollama.service
# Add: Environment="OLLAMA_KEEP_ALIVE=0"
```

---

## Using with VLA Code

### Example: VLA Policy with Local Llama

```python
from vla_policy_learner import VLAPolicyLearner
import requests
import json

class LocalVLAPolicy:
    def __init__(self, model="llama2"):
        self.model = model
        self.policy = VLAPolicyLearner()  # For fine-tuned action head

    def infer(self, image, instruction):
        """Infer action using local LLM + learned policy."""

        # Step 1: LLM analyzes scene (2-way)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": f"""
                Analyze this robot scene: [image description]
                Task: {instruction}

                Provide:
                1. Object detection
                2. Target position [x, y, z]
                3. Gripper force (10-300N)

                Response format: JSON
                """,
                "stream": False,
            }
        )

        llm_analysis = response.json()["response"]

        # Step 2: Parse and refine with learned policy
        action = json.loads(llm_analysis)

        # Step 3: Optional: Run through fine-tuned action head for refinement
        # refined_action = self.policy.infer(image, instruction)

        return action

# Usage
vla = LocalVLAPolicy(model="llama2")
action = vla.infer(robot_image, "Pick up the red cube")
robot.execute(action)
```

---

## Troubleshooting

### Problem 1: "Port 11434 already in use"

```bash
# Kill existing process
lsof -ti:11434 | xargs kill -9

# Or use different port
ollama serve --port 11435
```

### Problem 2: "CUDA out of memory"

```bash
# Use smaller model
ollama pull mistral  # 7B instead of 13B

# Or set GPU memory limit
CUDA_VISIBLE_DEVICES=0 ollama serve
```

### Problem 3: Very slow inference (CPU only)

```bash
# Check if GPU is being used
ollama logs

# If no GPU, install:
# - NVIDIA: https://developer.nvidia.com/cuda-downloads
# - AMD: https://rocmdocs.amd.com/
# - Apple: Already included (M1/M2/M3)
```

### Problem 4: Model keeps unloading

```bash
# Set environment variable to keep model loaded
export OLLAMA_KEEP_ALIVE=24h

# Or disable unloading
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

---

## Benchmarks: Local vs API

**Scenario**: Analyze robot scene image + predict action

| Approach | Cost | Speed | Setup |
|----------|------|-------|-------|
| **GPT-4 Vision (API)** | $0.03 | 3s | Easy |
| **Llama 2 7B (Local GPU)** | $0 | 0.5s | Medium |
| **Llama 2 7B (Local CPU)** | $0 | 5s | Medium |
| **Mistral 7B (Local GPU)** | $0 | 0.3s | Medium |

**For development**: Use local Mistral/Llama
**For production**: Use GPT-4 (better quality)
**For cost**: Local models (free after setup)
**For speed**: Local GPU (50-100× faster than API)

---

## Next Steps

1. **Install Ollama**: `brew install ollama` (Mac) or download from ollama.ai
2. **Pull a model**: `ollama pull mistral` (1-2 minutes, 4GB download)
3. **Start server**: `ollama serve` (runs on localhost:11434)
4. **Test it**: Use Python example above to verify it works
5. **Integrate into VLA**: Use in your policy or evaluation code

---

## Further Resources

- **Ollama**: https://ollama.ai/
- **LM Studio**: https://lmstudio.ai/
- **Llama 2**: https://llama.meta.com/
- **Mistral**: https://mistral.ai/
- **LLaVA** (Vision): https://github.com/haotian-liu/LLaVA

---

## Quick Reference

```bash
# Install
brew install ollama  # Mac

# Download model (choose one)
ollama pull mistral           # 7B, fast, good
ollama pull llama2            # 7B, default
ollama pull llama2:13b        # 13B, better quality
ollama pull llava             # Vision-language

# Start server
ollama serve

# Test (in another terminal)
curl http://localhost:11434/api/generate -d '{"model":"llama2","prompt":"Hello","stream":false}'

# Use in Python
import requests
resp = requests.post("http://localhost:11434/api/generate", json={"model":"llama2","prompt":"..","stream":false})
print(resp.json()["response"])
```

---

**Ready?** Install Ollama and try Exercise 2 in exercises.md!
