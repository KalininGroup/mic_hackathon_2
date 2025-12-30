import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"


def call_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 200
            }
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def explain_from_stats(stats, user_prompt):
    prompt = f"""
You are a biomedical imaging expert.

Segmentation results:
- Number of detected objects: {stats['count']}
- Mean object area (pixels): {stats['mean_area']}

User question:
{user_prompt}

Instructions:
- Explain the segmentation results
- Discuss density and morphology
- Avoid medical diagnosis
- State uncertainty explicitly

Answer:
"""
    return call_llm(prompt)
