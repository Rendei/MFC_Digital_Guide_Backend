from pydantic import BaseModel


class RequestModel(BaseModel):
    document_id: str
    user_request: str
    model_name: str = "Meta-Llama-3.3-70B-Instruct-Turbo"
