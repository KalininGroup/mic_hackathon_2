import speech_recognition as sr
import pyttsx3

def speak_text(command):
        engine=pyttsx3.init()
        voices = engine.getProperty('voices')
        #for idx, voice in enumerate(engine.getProperty('voices')):
        #     print(f"{idx} {voice}")
        voice_index = 3
        engine.setProperty('voice', voices[voice_index].id)
        engine.setProperty('volume', 1)
        engine.say(command)
        engine.runAndWait()
    
def listen_and_recognize(
        device_index=None,language="en-US",
        ambient_duration=0.3,timeout=5,
        phrase_time_limit=10,verbose=False
        )->str:
        r = sr.Recognizer()
        
        #for index, name in enumerate(sr.Microphone.list_microphone_names()):
        #    print(f"Device {index}: {name}")


        try:
            with sr.Microphone(device_index=device_index) as source:
                
                if verbose: print("🔊 Calibrating to ambient noise…")
                r.adjust_for_ambient_noise(source, duration=ambient_duration)
                if verbose: print(f"🎤 Say something (timeout={timeout}s, limit={phrase_time_limit}s)…")
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            #text = r.recognize_amazon(audio, language=language)
            text = r.recognize_google(audio, language=language)
            if verbose:
                 print(f'You said: "{text.lower()}"')
                 speak_text(f'You said: "{text.lower()}"')
            return text.lower()

        except sr.WaitTimeoutError:
            if verbose: print("⏱️ Timeout: no speech detected in the allocated time.")
            return "timeout"
        except sr.UnknownValueError:
            if verbose: print("🤷 Speech was not understood (low volume, noise, or unclear).")
            return "unclear"
        except sr.RequestError as e:
            if verbose: print(f"🌐 API/Network error with Google Speech: {e}")
            return "error"
        except OSError as e:
            if verbose: print(f"🎙️ Microphone error: {e} (check device index or permissions)")
            return "error"
        except Exception as e:
             if verbose: print(f"🚨 Fatal error: {e}")
             return None