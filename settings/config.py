from decouple import config

class Config:

    NUM_THREADS = config('NUM_THREADS', 50)

