import requests

def get_chatbot_response(prompt_with_context):
    url = "http://localhost:11434/api/generate"
    
    system_instruction = (
        "You are a Forensic AI Assistant. Use the provided facial emotion context "
        "to adjust your tone. If the user looks angry, be professional and de-escalating. "
        "If they look sad, be empathetic. Always provide forensic advice."
    )

    payload = {
        "model": "llama3",
        "prompt": f"{system_instruction}\n\nContext: {prompt_with_context}",
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        return {"response": response.json().get("response", "I am processing the data...")}
    except:
        return {"response": "I cannot reach the Llama3 server. Please ensure 'ollama run llama3' is active."}