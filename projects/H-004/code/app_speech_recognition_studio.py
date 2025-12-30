""" Have you always dreamed of talking to your AFM?
    Here you go...
    @author: Hans Gunstheimer (Nanosurf)
"""
#%%
import afm
import stt

if __name__ == "__main__":
    my_afm = afm.Afm()
    
    # Waits for microphone input
    while True:
        # Get microphone input
        result = stt.listen_and_recognize(
            device_index=2, language="en-US",
            ambient_duration=0.1, timeout=10,
            phrase_time_limit=5, verbose=True
            )

        # Fake LLM decides what to do with input:
        if result=="hey drive":
            print(f"Hey Hans, what can I do for you today?")
            stt.speak_text(f"Hey Hans my love, what can I do for you today?")

        elif result=="approach":
            my_afm.connect_to_afm(verbose=False)
            my_afm.approach(verbose=True)

        elif result=="start imaging":
            my_afm.connect_to_afm(verbose=False)
            my_afm.start_imaging(verbose=True)
        
        elif result=="adjust laser":
            my_afm.connect_to_afm(verbose=False)
            my_afm.start_laser_align(verbose=True)

        elif result=="retract":
            my_afm.connect_to_afm(verbose=False)
            my_afm.retract(verbose=True)

        elif result=="stop imaging":
            my_afm.connect_to_afm(verbose=False)
            my_afm.stop_imaging(verbose=True)
        
        elif result=="stop":
            my_afm.connect_to_afm(verbose=False)
            my_afm.stop(verbose=True)
        
        elif result=="timeout":
            pass

        elif result=="unclear":
            print(f"I could not understand you. Please try again.")
            stt.speak_text(f"I could not understand you. Please try again.")

        elif result=="error":
            print(f"A major error has occured")
            stt.speak_text(f"A major error has occured. Please call customer support.")

        elif result=="exit":
            print(f"Exiting Alexa for AFM")
            stt.speak_text(f"Exiting Alexa for AFM. See you next time.")
            break

        else:
            print(f"Command {result} is not supported.")
            stt.speak_text(f"Command {result} is not supported.")


# %%