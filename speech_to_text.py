import speech_recognition as sr

def speech_to_text_and_save():
    print("Select a language:")
    print("1. Thai")
    print("2. English")
    language_choice = input("Enter 1 for Thai or 2 for English: ")

    if language_choice == "1":
        language_code = "th-TH"
        language_name = "Thai"
    elif language_choice == "2":
        language_code = "en-US"
        language_name = "English"
    else:
        print("Invalid choice. Exiting.")
        return

    recognizer = sr.Recognizer()

    filename = f"output_{language_name.lower()}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        while True:
            print(f"Adjusting microphone... Please speak in {language_name}:")
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source)

            try:
                print("Converting speech to text...")
                text = recognizer.recognize_google(audio_data, language=language_code)
                print(f"Text ({language_name}): {text}")

                file.write(f"Text ({language_name}):\n{text}\n")

            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Error connecting to Google API: {e}")

            continue_choice = input("Do you want to continue recording? (y/n): ")
            if continue_choice.lower() != 'y':
                print(f"Text has been saved to '{filename}'")
                break

if __name__ == "__main__":
    speech_to_text_and_save()
