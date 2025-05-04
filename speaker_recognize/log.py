import logging
logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("speaker_embedding.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s -%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)