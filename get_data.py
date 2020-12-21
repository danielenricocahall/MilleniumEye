import os
from concurrent.futures.thread import ThreadPoolExecutor
import requests

import structlog as structlog
from requests import HTTPError

from settings.config import Config

_LOGGER = structlog.get_logger(__name__)


def get_all_yugioh_cards():
    if not os.path.exists('./cards'):
        os.mkdir('./cards')
    try:
        _LOGGER.info("Started loading cards! This may take a few minutes...")
        response = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')
        response.raise_for_status()
        cards = response.json()['data']
        with ThreadPoolExecutor(Config.NUM_THREADS) as executor:
            executor.map(save_card, cards)
        _LOGGER.info("Finished loading cards!")
    except HTTPError as e:
        _LOGGER.error(str(e))


def save_card(card):
    card_img_url = card['card_images'][0]['image_url']
    name = card['name']
    name = name.replace('/', '_')
    img_data = requests.get(card_img_url)
    img_data.raise_for_status()
    with open(f'./cards/{name}.png', 'wb') as handler:
        handler.write(img_data.content)
    _LOGGER.info(f"Loaded {name}")


if __name__ == "__main__":
    get_all_yugioh_cards()
