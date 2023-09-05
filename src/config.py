import os

RAW_DATA_FILENAME = 'raw_paperdigest_data.csv'
TITLES_PER_PROMPT = 10

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
