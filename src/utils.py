import logging
import time

import requests

import config


def prompt_chat_gpt(system_prompt, user_prompt, max_tokens):
    """Prompt ChatGPT with the given prompts and token limit.

    Parameters
    ----------
    system_prompt : str
        The system-level prompt.
    user_prompt : str
        The user prompt.
    max_tokens : int
        The max tokens to return.

    Returns
    -------
    response : str
        The raw response in string format.
    """
    acc = 0
    response = None
    while True:
        if acc > 5:
            raise ConnectionRefusedError('Model is overloaded.')
        req = {
            "model": "gpt-3.5-turbo-1106",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        try:
            response = requests.post(config.OPENAI_URL, headers=config.OPENAI_HEADERS, json=req, timeout=60)
        except requests.exceptions.ReadTimeout:
            logging.warning('Request timed out. Pausing and trying again.')
            time.sleep(3)
            acc += 1
            continue
        try:
            response = response.json()["choices"][0]["message"]["content"]
            break
        except KeyError:
            if response.status_code in (429, 503):
                # TODO: This is not sufficient. Status code 429 is used for multiple things.
                logging.warning('OpenAI API is overloaded. Pausing before trying again.')
                time.sleep(3)
                acc += 1
            else:
                logging.error(f'Got response code {response.status_code} with response {response.json()}')
                raise ConnectionRefusedError(f'Response code {response.status_code}')

    if response is None:
        raise ConnectionError('Maximum retries reached.')
    return response
