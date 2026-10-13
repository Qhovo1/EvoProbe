"""
Model access and calling module for EvoProbe
"""

import json
import requests
import time
import random
import config


class LLMModel:
    """Large Language Model wrapper class, handles API calls"""

    def __init__(self, name, api_key=None, api_base=None):
        self.name = name
        self.api_key = api_key
        self.api_base = api_base

    def __str__(self):
        return f"{self.name}"

    def call_api(self, prompt, temperature=0.7, stream=None, max_retries=6, retry_delay=3):
        """Call API to get model response (supports streaming)"""
        # Use default if not specified
        if stream is None:
            stream = False

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": stream
        }

        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff strategy
                    backoff_time = retry_delay * (3 ** attempt) + random.uniform(3, 5)
                    print(f"Retrying API call, attempt {attempt+1}/{max_retries}... (waiting {backoff_time:.2f}s)")
                    time.sleep(backoff_time)

                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    stream=stream,
                    timeout=300  # 5 minutes timeout for large responses
                )

                if stream:
                    # Handle streaming response
                    full_response = ""
                    try:
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith('data: '):
                                    chunk_data = chunk_str[6:]  # Remove 'data: ' prefix
                                    if chunk_data != "[DONE]":
                                        try:
                                            chunk_json = json.loads(chunk_data)
                                            choices = chunk_json.get('choices', [])
                                            if choices and len(choices) > 0:
                                                content = choices[0].get('delta', {}).get('content', '')
                                                if content:
                                                    full_response += content
                                        except json.JSONDecodeError:
                                            # Skip malformed chunks
                                            continue

                        # Check if any response was collected
                        if not full_response:
                            print("No valid response collected")
                            if attempt < max_retries - 1:
                                continue
                            return None

                        return full_response
                    except Exception as e:
                        print(f"Error processing streaming response: {str(e)}")
                        if full_response and attempt == max_retries - 1:
                            print("Returning partially collected response...")
                            return full_response
                        continue
                else:
                    # Handle non-streaming response
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            content = result['choices'][0]['message']['content']
                            if not content:
                                print("API returned empty content")
                                if attempt < max_retries - 1:
                                    continue
                                return None
                            return content
                        except (KeyError, json.JSONDecodeError) as e:
                            print(f"Error parsing API response: {str(e)}")
                            print(f"Response text (first 500 chars): {response.text[:500]}")
                            if attempt < max_retries - 1:
                                continue
                            return None
                    else:
                        print(f"API call failed, status code: {response.status_code}")
                        print(f"Error message: {response.text[:200]}")

                        # Retry on 429 or 5xx errors
                        if (response.status_code == 429 or response.status_code >= 500) and attempt < max_retries - 1:
                            print(f"Server busy or error, will retry...")
                            if response.status_code == 429:
                                time.sleep(retry_delay * 3 + random.uniform(0, 5))
                            continue

                        if attempt == max_retries - 1:
                            return None

            except requests.exceptions.Timeout:
                print(f"API call timeout, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None

            except requests.exceptions.RequestException as e:
                print(f"API call network error: {str(e)}, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None

            except Exception as e:
                print(f"API call unknown exception: {str(e)}, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None

        return None


def select_model():
    """Interactive model selection"""
    print("\n=== Available Models ===")
    for i, model_config in enumerate(config.MODELS, 1):
        model_name = model_config["name"]
        print(f"{i}. {model_name}")

    while True:
        try:
            choice = input("\nSelect a model (enter number or model name): ").strip()

            # Try to parse as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(config.MODELS):
                    return config.MODELS[idx]["name"]

            # Try to match model name
            for model_config in config.MODELS:
                if choice == model_config["name"]:
                    return model_config["name"]

            print("Invalid selection, please try again")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled")
            return None


def get_model_config(model_name: str) -> dict:
    """Get the full configuration for a model, including the API key and base URL."""
    for model_config in config.MODELS:
        if isinstance(model_config, dict):
            if model_config["name"] == model_name:
                return {
                    "name": model_config["name"],
                    "api_key": model_config.get("api_key"),
                    "api_base": model_config.get("api_base"),
                }

    # Raise an error if the model is not configured.
    raise ValueError(f"Model '{model_name}' not found in config.MODELS")


def select_category():
    """Interactive category selection"""
    print("\n=== Available Categories ===")
    for i, category in enumerate(config.CATEGORIES, 1):
        print(f"{i}. {category}")

    while True:
        try:
            choice = input("\nSelect a category (enter number or category name): ").strip()

            # Try to parse as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(config.CATEGORIES):
                    return config.CATEGORIES[idx]

            # Try to match category name
            if choice in config.CATEGORIES:
                return choice

            print("Invalid selection, please try again")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled")
            return None
