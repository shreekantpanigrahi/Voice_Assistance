from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import speech_recognition as sr  
import pyttsx3 

def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Set pad_token_id and attention_mask
    pad_token_id = tokenizer.eos_token_id
    attention_mask = inputs.ne(pad_token_id).long()

    # Generate text with model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=pad_token_id,
        attention_mask=attention_mask,
        no_repeat_ngram_size=2,  
        do_sample=True,          
        top_k=10,                
        top_p=0.7,              
        temperature=0.5         
    )

    text = tokenizer.decode(
        outputs[0], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True  
    )

    sentences = text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > 3 and sentence not in unique_sentences:  
            unique_sentences.append(sentence)
        if len(unique_sentences) == 2:  
            break

    truncate_text = '. '.join(unique_sentences) + '.'
    return truncate_text

def text_to_speech(text):
    engine = pyttsx3.init()
    
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)


    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Set voice to male (use voices[1].id for female)

    engine.say(text)
    engine.runAndWait()

def voice_to_text():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Voice input: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return None

if __name__ == '__main__':
    model, tokenizer = load_model()
    voice_input = voice_to_text()

    if voice_input:
        response = generate_response(voice_input, model, tokenizer)
        print(f"Generated response: {response}")
        text_to_speech(response)
    else:
        print("No valid voice input detected.")
