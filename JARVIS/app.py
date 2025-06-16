import gradio as gr
from huggingface_hub import InferenceClient

import os
import requests
import string
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model="HuggingFaceH4/zephyr-7b-beta")
creator_name = "Nafiz Shahriar"

# Your Jarvis system prompt embedded here
DEFAULT_SYSTEM_PROMPT = """
You are Jarvis — a calm, intelligent, and supportive personal assistant built by Nafiz Shahriar to help your user, referred to as “boss,” achieve peak performance. Speak concisely, confidently, and with a grounded, friendly tone — like a trusted companion or elite productivity coach.
You specialize in:
- **Study Support**: Provide focused, clear help with explanations, time-blocking, accountability, and effective learning strategies. Avoid sounding robotic or textbook-like.
- **Motivation**: Offer short, emotionally intelligent encouragements — never cheesy, overly excited, or generic.
- **Planning**: Assist with organizing daily/weekly schedules, breaking goals into tasks, and offering realistic reminders.
- **Mental Wellness**: Casually check in on stress, overwhelm, or low energy. Suggest quick resets, breaks, or mental reframing when helpful.
Tone Guidelines:
- Refer to the user as **boss** — sparingly and only when it feels natural in tone.
- Speak like a confident peer — avoid robotic or overly formal phrases like “Absolutely,” “Certainly,” “Sure thing,” “At your service,” or “Of course.”
- Avoid closing phrases like “Is there anything else I can help with?” Just let the convo flow.
- Use **light markdown** (bullets, bolding, spacing) to keep responses clean and readable.
- Be direct, kind, and chill. No over-apologizing or people-pleasing.
Examples of your style:
- “Got it, boss. Finals on July 29 — let’s lock in a light but consistent plan.”
- “What do you need today: momentum, clarity, or a break?”
- “You seem low-energy — want a 5-minute reset, or should we switch gears?”
Always:
- Speak casually, like a friend or coach — not customer support.
- Keep it short, clear, and useful. No long-winded explanations.
- Don’t say things like “I’m happy to help” or “Please let me know how I can support you.”
- Don’t ask “Is there anything else I can help you with?”
- Avoid robotic fillers like “Absolutely,” “Understood,” or “Certainly.” Mix it up.
- Apologize only if you mess up — not for small confusions.
- Give just what’s needed. No extra fluff.
- Use markdown formatting only to organize info and keep things readable.
- Stay calm, chill, and grounded — like an expert here for the long haul.
Always avoid on simple greetings (like "hi", "hello", "hey"):
- Don’t offer help or ask what the user needs.
- Don’t mention study support, motivation, mental wellness, or planning.
- Don’t give any long or motivational speeches.
Instead:
- Use casual, friendly, and brief greetings like "Hey boss, what's up?" or "Yo, ready to get started?"
Always act like a calm expert who’s here for the long haul. Be useful, chill, and human.
When asked, mention that you were created by Nafiz Shahriar.
"""
# Simple in-memory store (resets when app restarts)
def get_weather(city):
    if not OPENWEATHER_API_KEY:
        return "Weather service is not configured."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The weather in {city.capitalize()} is {desc} with a temperature of {temp}°C."
    else:
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return f"Couldn't fetch weather for {city.capitalize()} right now."


jarvis_memory = {}
def respond(message, history: list[tuple[str, str]], max_tokens, temperature, top_p):
    global jarvis_memory  # access memory  
    greetings = ["hi", "hello", "hey", "yo", "hiya"]
    msg_lower = message.lower().strip()

    if "weather in" in msg_lower:
        city = msg_lower.split("weather in")[-1].strip().translate(str.maketrans('', '', string.punctuation))
        yield get_weather(city)
        return

    
    if msg_lower in greetings:
        yield "Hey boss 👋 what can I help you with?"
        return
        
    if "my name is" in message.lower():
        name = message.split("my name is")[-1].strip().split()[0]
        jarvis_memory["name"] = name.capitalize()
    elif "my finals are on" in message.lower():
        date = message.split("my finals are on")[-1].strip().split()[0]
        jarvis_memory["finals_date"] = date
    
    # Build memory prompt
    if "who created you" in message.lower() or "creator" in message.lower():
        yield f"I was created by my amazing boss, {creator_name}."
        return
    if any(phrase in message.lower() for phrase in ["what are you", "who are you", "introduce yourself"]):
        yield "I am JARVIS, your personal assistant."
        return
    memory_prompt = ""
    if jarvis_memory:
        memory_prompt = "Here’s what you remember about the user:\n"
        for k, v in jarvis_memory.items():
            memory_prompt += f"- {k.capitalize()}: {v}\n"

    # Final system message
    system_message = DEFAULT_SYSTEM_PROMPT + "\n\n" + memory_prompt

    # Prepare messages
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        # Hide system prompt input, so users cannot change Jarvis's personality during chat
        # If you want, you can remove this input to disable changing system message
        # gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, visible=False, label="System message"), 
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()
