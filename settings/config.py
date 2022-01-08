from decouple import config

class Config:

    NUM_THREADS = config('NUM_THREADS', 50)

    YUGIOH_DATA_URL = config('YUGIOH_DATA_URL', 'https://db.ygoprodeck.com/api/v7/cardinfo.php')

