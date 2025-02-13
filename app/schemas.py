from pydantic import BaseModel


class RequestModel(BaseModel):
    document_id: str
    user_request: str
