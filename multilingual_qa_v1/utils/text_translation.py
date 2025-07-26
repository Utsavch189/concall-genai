import google.generativeai as genai
import os
from dotenv import load_dotenv
import openai

    
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")
openai.api_key = os.getenv('OPENAI_API_KEy')

def count_tokens(model, text):
    return model.count_tokens(text).total_tokens

language_codes = {
    "aa": "Afar",
    "ab": "Abkhazian",
    "ae": "Avestan",
    "af": "Afrikaans",
    "ak": "Akan",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bi": "Bislama",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ce": "Chechen",
    "ch": "Chamorro",
    "co": "Corsican",
    "cr": "Cree",
    "cs": "Czech",
    "cu": "Church Slavic",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "ff": "Fula",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hz": "Herero",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "ii": "Yi",
    "ik": "Inupiat",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "kl": "Greenlandic",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "quc": "K'iche'",
    "rm": "Romansh",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "sg": "Sango",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "st": "Sotho",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Tswana",
    "to": "Tongan",
    "tr": "Turkish",
    "ts": "Tswana",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "zu": "Zulu"
}

def translate_text_v1(text:str, target_language="en"):

    target_language = language_codes[target_language]

    prompt = f"""
        System: You are a professional translator. Your task is to:
        - Detect the source language of the given text.
        - Translate the text into {target_language} with some desi {target_language} flavour just like we use to say generally, preserving:
          - The original sentence structure and formatting.
          - Technical and financial terms (e.g., EBITDA, P/E, P/B, Gross Margin) by adding them in **parentheses** immediately after their translated equivalent.
        
        Examples:
          English: "The P/E ratio indicates price over earnings."
          -> "[Translated text] (P/E)"
        
        Return only the translated text. Do not include any explanations, notes, or metadata.
        
        If the source text is already in English and the target language is also English, skip translation and return the original text as is.
        
        Text to translate:
        \"\"\"
        {text.strip()}
        \"\"\"
        """
    
    response = model.generate_content(prompt)
    result = response.text.strip()
    prompt_tokens = count_tokens(model, prompt)
    response_tokens = count_tokens(model, result)

    return {
        "translated_text":result,
        "token_usage":{
            "total_token":prompt_tokens+response_tokens,
            "input_token":prompt_tokens,
            "output_token":response_tokens
        }
    }

def translate_text_v2(text:str, target_language="en"):

    target_language = language_codes[target_language]

    prompt = f"""
        System: You are a professional translator. Your task is to:
        - Detect the source language of the given text.
        - Translate the text into {target_language}, preserving:
          - The original sentence structure and formatting.
          - Technical and financial terms (e.g., EBITDA, P/E, P/B, Gross Margin) by adding them in **parentheses** immediately after their translated equivalent.
        
        Examples:
          English: "The P/E ratio indicates price over earnings."
          -> "[Translated text] (P/E)"
        
        Return only the translated text. Do not include any explanations, notes, or metadata.
        
        If the source text is already in English and the target language is also English, skip translation and return the original text as is.
        
        Text to translate:
        \"\"\"
        {text.strip()}
        \"\"\"
        """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0
    )
    usage = response.usage
    return {"translated_text": response.choices[0].message.content.strip(),"token_usage":{
        "total_token":usage.total_tokens,
        "input_token":usage.prompt_tokens,
        "output_token":usage.completion_tokens
    }}

if __name__ == "__main__":
    # res = translate_text_v2(
    #     text="company tar agar 2 bochor er revenue and growth comparison dekhao !",
    # )
    # print(res)
    pass