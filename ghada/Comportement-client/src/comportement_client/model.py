from inference_sdk import InferenceHTTPClient

from .config import Settings


def load_model(settings: Settings) -> InferenceHTTPClient:

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=settings.roboflow_api_key,
    )

    return client

def detect(client, frame, settings):

    result = client.infer(
        frame,
        model_id=settings.model,
    )

    return result