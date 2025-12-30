import nanosurf
import stt
import time

class Afm:
    def __init__(self):
        self.studio = nanosurf.Studio()
        self.connect_to_afm(verbose=True)
    

    def approach(self, verbose=False):
        self.studio.spm.workflow.approach.start_approach()
        if verbose:
            stt.speak_text(f"Connecting to DriveAFM")
    
    def connect_to_afm(self,verbose=False):
        self.studio.connect()
        if verbose:
            print(f"Available sessions: {self.studio.main.session.list()}")
            print(f"Connected with session '{self.studio.session_id}'")
            stt.speak_text("Connecting to DriveAFM")
    
    def start_laser_align(self, verbose=False):
        self.studio.spm.workflow.laser_auto_align.auto_align_all()
        if verbose:
            stt.speak_text(f"DriveAFM starts laser alignment")
    
    def retract(self, verbose=False):
        self.studio.spm.workflow.approach.start_retract()
        if verbose:
            stt.speak_text(f"Start retracting from surface")
        time.sleep(5)
        self.studio.spm.workflow.approach.abort()

    def start_imaging(self, verbose=False):
        self.studio.spm.workflow.imaging.start_imaging()
        if verbose:
            stt.speak_text(f"DriveAFM starts imaging")
    
    def stop(self, verbose=False):
        self.studio.spm.workflow.approach.abort()
        if verbose:
            stt.speak_text(f"DriveAFM stops approaching.")

    def stop_imaging(self, verbose=False):
        self.studio.spm.workflow.imaging.stop_imaging()
        if verbose:
            stt.speak_text(f"DriveAFM stops imaging")