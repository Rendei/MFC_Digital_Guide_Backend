import json
import time
from fastapi import HTTPException
from app.data_loader import load_documents
from app.utils import clean_and_format_text, join_strings_in_dict
from openai import OpenAI

from app.config import load_config
from app.metrics.calculate_roadmap_metrics import evaluate_text
# from metrics.calculate_roadmap_metrics import evaluate_text

config = load_config()
api_key = config.get("api_key")


client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

documents = load_documents()
documents = join_strings_in_dict(documents)


def generate_roadmap_batch(document_id: str, user_request: str, model_name="Meta-Llama-3.3-70B-Instruct-Turbo"):
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    document_text = documents[document_id]
    requests = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"klusterai/{model_name}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, that provides users with step by step instructions",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Тебе предоставляется текст документа, необходимо внимательно прочитать этот текст. "
                            f"Составь дорожную карту (список задач), чтобы получить то, что хочет пользователь. "
                            f"Запрос пользователя: {user_request}, Текст документа: {document_text}"
                        ),
                    },
                ],
                "max_completion_tokens": 1000,
            },
        },
    ]

    file_name = "batch_input.jsonl"
    with open(file_name, "w") as file:
        for request in requests:
            file.write(json.dumps(request) + "\n")

    try:
        batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
        batch_request = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        while True:
            batch_status = client.batches.retrieve(batch_request.id)
            if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                break
            time.sleep(10)  # Check every 10 seconds

        if batch_status.status.lower() == "completed":
            result_file_id = batch_status.output_file_id
            raw_results = client.files.content(result_file_id).content
            try:
                results_json = json.loads(raw_results.decode("utf-8"))
                road_map_text = results_json["response"]["body"]["choices"][0]["message"]["content"]
                cleaned_text = clean_and_format_text(road_map_text)
                metrics = evaluate_text(cleaned_text, document_text)

                return cleaned_text, metrics
            except (KeyError, json.JSONDecodeError) as e:
                raise HTTPException(status_code=500, detail=f"Error processing the response: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Batch failed with status: {batch_status.status}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_roadmap_livetime(document_id: str, user_request: str):
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    document_text = documents[document_id]

    chat_completion = client.chat.completions.create(
        model="klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
        messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, that provides users with step by step instructions",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Тебе предоставляется текст документа, необходимо внимательно прочитать этот текст. "
                            f"Составь дорожную карту (список задач), чтобы получить то, что хочет пользователь. "
                            f"Запрос пользователя: {user_request}, Текст документа: {document_text}"
                        ),
                    },
                ],
    )

    chat_completion_json = chat_completion.to_dict()

    #try:
    road_map_text = chat_completion_json["choices"][0]["message"]["content"]
    cleaned_text = clean_and_format_text(road_map_text)
    return cleaned_text
    #except (KeyError, json.JSONDecodeError) as e:
        #raise HTTPException(status_code=500, detail=f"Error processing the response: {str(e)}")