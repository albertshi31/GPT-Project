from transformers import BertTokenizer, BertForSequenceClassification
import torch
import speech_recognition as sr
from gtts import gTTS
import os
import platform
import webbrowser
import openai

# Set up the API key for OpenAI
openai.api_key = 'Hidden'

# Rick Roll!
keyword = "banana"
label_to_intent = {
    0: "Query",
    1: "Command",
    2: "Information",
    3: "Request",
    4: "Feedback",
    5: "Greeting"
}

# Function to setup BERT model
def setup_models(bert_model_name='bert-base-uncased'):
    # Setup BERT model and tokenizer for intent classification
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=len(label_to_intent))

    return bert_tokenizer, bert_model

# Function to generate response using GPT-3.5
def generate_response(prompt, tokenizer_bert, model_bert):
    # Tokenize input for BERT
    inputs = tokenizer_bert(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Perform intent classification with BERT
    with torch.no_grad():
        outputs = model_bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Determine intent from logits
    intent_label = torch.argmax(logits, dim=-1).item()
    intent_name = label_to_intent.get(intent_label, "Unknown")

    # Print logits for debugging
    print(f"Logits: {logits}")
    print(f"Predicted Intent: {intent_name}")

    # Use GPT-3.5 to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the response text
    response_text = response['choices'][0]['message']['content'].strip()

    return response_text

# Function to recognize speech from microphone
def recognize_speech_from_mic(recognizer, mic):
    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        transcript = recognizer.recognize(audio)
        print(f"You said: {transcript}")
        return transcript
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == "__main__":
    # Setup models
    bert_tokenizer, bert_model = setup_models()

    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        while True:
            # Recognize speech input
            prompt = recognize_speech_from_mic(recognizer, mic)

            if prompt:
                # Check for keyword
                if keyword in prompt.lower():
                    webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                else:
                    # Generate response
                    response = generate_response(prompt, bert_tokenizer, bert_model)
                    print(f"GPT-3.5 says: {response}")

                    # Convert response to speech
                    tts = gTTS(text=response, lang='en')
                    tts.save("GPToutput.mp3")

                    # Play speech based on OS
                    system = platform.system()
                    if system == "Windows":
                        os.system("start GPToutput.mp3")
                    elif system == "Darwin":  # macOS
                        os.system("afplay GPToutput.mp3")
                    elif system == "Linux":
                        os.system("aplay GPToutput.mp3")

    except KeyboardInterrupt:
        print("\nExiting...")
