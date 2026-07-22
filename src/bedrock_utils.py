import json
import logging
import time
import traceback

import boto3.session as boto3_session
import botocore.config


def extract_xml_tag(generation: str, tag):
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return
    value = generation[begin:end].strip()
    return value


def predict_one_eg_mistral(x):
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:latest",
        "prompt": x["prompt_input"],
        "stream": False,
        "options": {
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 10,
            "num_predict": 512
        }
    }
    
    success = False
    response = None
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                success = True
                break
        except Exception as e:
            logging.error(f"Ollama attempt {i+1} failed: {e}")
            time.sleep(2)

    if success:
        response_body = response.json()
        logging.info(response_body.get("response", ""))
        return response_body.get("response", "")
    else:
        return ""


def predict_one_eg_claude_instant(x):
    import requests
    url = "http://localhost:11434/api/generate"
    prompt = "Human: {prompt}\nWrite your summary in <summary> XML tags.\n\nAssistant: ".format(
        prompt=x["prompt_input"].strip()
    )
    payload = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 10,
            "num_predict": 512
        }
    }

    success = False
    response = None
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                success = True
                break
        except Exception as e:
            logging.error(f"Ollama attempt {i+1} failed: {e}")
            time.sleep(2)

    if success:
        response_body = response.json()
        generation = response_body.get("response", "")
        summary = extract_xml_tag(generation, "summary")
        logging.info(summary or generation)
        return summary or generation
    else:
        return ""
