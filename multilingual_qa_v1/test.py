from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEy'))

with open("audio.ogg", "rb") as audio_file:
    translate = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )

print(translate)