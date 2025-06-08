import time

class Config:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = None
        self.current_time = None
        self.spend_time = None

    def update_current_time(self):
        self.current_time=time.time()

    def get_spend_time(self):
        self.current_time = time.time()
        if self.last_time is None:
            self.spend_time = self.current_time - self.start_time
        else:
            self.spend_time = self.current_time - self.last_time
        
        self.last_time = self.current_time
        return int(self.spend_time)
