from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForSequenceClassification
import torch
import speech_recognition as sr
from gtts import gTTS
import os
import platform
import webbrowser

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

# Function to setup BERT and GPT-2 models
def setup_models(gpt_model_name='gpt2-large', bert_model_name='bert-base-uncased'):
    # Setup GPT-2 model and tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
    gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    # Setup BERT model and tokenizer for intent classification
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=len(label_to_intent))

    return gpt_tokenizer, gpt_model, bert_tokenizer, bert_model

# Function to generate response
def generate_response(prompt, tokenizer_gpt, model_gpt, tokenizer_bert, model_bert):
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

    # Generate response using GPT-2
    inputs_gpt = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids_gpt = inputs_gpt["input_ids"]
    attention_mask_gpt = inputs_gpt["attention_mask"]

    with torch.no_grad():
        outputs_gpt = model_gpt.generate(
            input_ids_gpt, 
            attention_mask=attention_mask_gpt, 
            max_length=150, 
            num_return_sequences=1, 
            num_beams=2,  # Set number of beams for beam search
            no_repeat_ngram_size=2, 
            early_stopping=True,
            pad_token_id=tokenizer_gpt.eos_token_id,
            top_k=50,  # Add top-k sampling for diversity
        )

    response = tokenizer_gpt.decode(outputs_gpt[0], skip_special_tokens=True).strip()

    # Remove the input prompt from the start of the response if it is present
    if response.lower().startswith(prompt.lower()):
        response = response[len(prompt):].strip()

    return response


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
    gpt_tokenizer, gpt_model, bert_tokenizer, bert_model = setup_models()

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
                    response = generate_response(prompt, gpt_tokenizer, gpt_model, bert_tokenizer, bert_model)
                    print(f"GPT-2 says: {response}")

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

