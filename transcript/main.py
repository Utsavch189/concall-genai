from openai import OpenAI
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEy'))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-pro")

def transcribe_audio():
    audio_file_path = "transcript/call.mpeg"

    with open(audio_file_path, "rb") as f:
        translate = client.audio.translations.create(
            model="whisper-1",
            file=f
        )

    print(translate.text)

    prompt = f"""
        You are a text processor. I will provide you a raw transcribed conversation. Your tasks are:

        1. Identify the different speakers and label them consistently as "Speaker 1", "Speaker 2", etc.
        2. Arrange the conversation **speaker-wise**, preserving the original order of statements.
        3. Translate the content into **Bengali** while keeping it natural and readable.
        4. Output in this exact format:

        Speaker 1 [with name if has any]: [text]
        Speaker 2 [with name if has any]: [text]
        Speaker 1 [with name if has any]: [text]
        ...

        Raw transcript:

        \"\"\"
        {translate.text}
        \"\"\"
    """

    response = model.generate_content(prompt)
    result = response.text.strip()

    print("Translated ----- > ")
    print()
    print(result)

if __name__ == "__main__":
    transcribe_audio()