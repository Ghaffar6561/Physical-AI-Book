# Setup Guide: Large Language Models for Robotics

This guide provides instructions for setting up and running Large Language Models (LLMs) locally for use in your robotics projects. Running models locally is crucial for applications requiring low latency, data privacy, and high customization.

## Option 1 (Recommended): Ollama

**Ollama** is a fantastic tool that dramatically simplifies the process of downloading, setting up, and running popular open-source LLMs. It bundles the model weights, configuration, and a server into a single, easy-to-use package.

### 1. Installation

*   **Linux & WSL**: Run the following command in your terminal:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
*   **macOS & Windows**: Download the appropriate installer from the [Ollama website](https://ollama.com/).

### 2. Running a Model

Once Ollama is installed, running a model is as simple as a single command.

```bash
# Pull and run the Mistral 7B model (a good balance of performance and size)
ollama run mistral

# Pull and run Llama 2 7B
ollama run llama2

# For a smaller, faster model (good for testing)
ollama run tinyllama
```

After you run this command, Ollama will download the model and start a local server. You can then interact with it directly in the terminal.

### 3. Using the API

Ollama also exposes a local REST API that you can use to interact with the model from your Python code (e.g., from your ROS 2 nodes). The API server runs at `http://localhost:11434`.

Here's a simple Python example using the `requests` library:

```python
import requests
import json

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        # The actual response is a JSON object within the 'response' key
        return json.loads(response.text)['response']
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == '__main__':
    my_prompt = "What is the capital of France?"
    llm_response = query_ollama(my_prompt)
    if llm_response:
        print(llm_response)
```

## Option 2: OpenAI API (Commercial)

If you want to use a state-of-the-art commercial model like GPT-4 for prototyping, you can use the OpenAI API.

### 1. Installation

Install the official OpenAI Python library:
```bash
pip install openai
```

### 2. API Key Setup

1.  Create an account on the [OpenAI Platform](https://platform.openai.com/).
2.  Go to your API Keys section and create a new secret key.
3.  **IMPORTANT**: Do not hardcode this key in your code. Set it as an environment variable.
    ```bash
    export OPENAI_API_KEY='your-secret-api-key'
    ```

### 3. Using the API in Python

```python
from openai import OpenAI

client = OpenAI() # The client automatically reads the OPENAI_API_KEY from your environment

def query_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", # Or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    my_prompt = "Explain the concept of reinforcement learning in simple terms."
    llm_response = query_gpt(my_prompt)
    if llm_response:
        print(llm_response)
```

## Prompt Engineering Best Practices

*   **Be Specific**: Clearly state the role, context, and desired output format.
*   **Use Few-Shot Examples**: Provide 2-3 examples of good inputs and outputs to guide the model.
*   **Use Delimiters**: Use characters like `###` or `---` to separate different parts of your prompt (e.g., instructions, examples, user input).
*   **Iterate**: Getting the prompt right is an iterative process. Start simple, see what the model produces, and refine your prompt to fix errors.
