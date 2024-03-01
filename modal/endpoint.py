## # HTTP Endpoint on Modal
##
## An endpoint on Modal is a fastest way to deploy a model for inference with Flywheel. You can also use
## the endpoint to test and experience the benefits of Flywheel.
##
## An endpoint is a RESTful API that accepts HTTP requests for inference and returns responses.
## In this example, we'll build a FastAPI based endpoint with Flywheel on Modal.
## With Flywheel, you get [unparalleled throughput and latency](https://mk1.ai/blog/flywheel-launch)
## for serving asynchronous inference requests (such as real-time chatbots).
##
## ## Endpoint Application
##
## The first step in the building the REST application is to define the input and output messages.
## For this example we'll use `pydantic` to define the input and output types to match the [MK1 Flywheel API](#mk1_api).
##
## The next step is to implement the endpoint using FastAPI. We'll run the endpoint on a barebones Debian
## container, as we only need this container to run the FastAPI application and to forward the generation
## requests to the MK1 Flywheel container.
##
## In this example we're setting the `keep_warm` option to 1, which means that at least container will be kept
## warm at all times. This is useful for low-latency applications, as it ensures that the container is always
## ready to serve requests. You can find more information about this topic in the [Modal documentation](https://modal.com/docs/guide/cold-start).
##
## The FastAPI application will support the following endpoints:
##
## - `/health`: A health check endpoint that returns a 503 status code if there are no runners available, and a
##   200 status code otherwise.
## - `/stats`: An endpoint that returns the current stats of the MK1 Flywheel container.
## - `/generate`: An endpoint that accepts a JSON payload with the generation request and returns the generation
##   response.
##
## This examples uses a [pre-baked](https://docs.mk1.ai/modal/configuration.html#list-of-pre-baked-images) Mistral-7b-instruct model.
## However, you can modify the example to use a supported model of your choice with [Bring-Your-Own-Model](https://docs.mk1.ai/modal/byom.html) (BYOM),
## where a volume preloaded with your model (perhaps, fine-tune) is used to setup the `Model` class.
##
## The example endpoint can be served with `modal serve endpoint.py`.

import modal

from typing import List
from pydantic import BaseModel


class GenerationRequest(BaseModel):
    text: str
    max_tokens: int
    eos_token_ids: List[int] = []
    max_input_tokens: int = 0
    num_samples: int = 1
    stop: List[str] = []
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class GenerationResponseSample(BaseModel):
    text: str
    generated_tokens: int
    finished: float
    finish_reason: str


class GenerationResponse(BaseModel):
    created: float
    finished: float
    num_samples: int
    prompt: str
    prompt_tokens: int
    responses: List[GenerationResponseSample]


stub = modal.Stub(
    "mk1-endpoint-backend",
    image=modal.Image.debian_slim(),
)


@stub.function(
    keep_warm=1,
    allow_concurrent_inputs=1024,
    timeout=600,
)
@modal.asgi_app(label="mk1-chat-endpoint")
def app():
    import modal
    import fastapi
    import fastapi.staticfiles

    web_app = fastapi.FastAPI()
    Model = modal.Cls.lookup(
        "mk1-flywheel-latest-mistral-7b-instruct", "Model", workspace="mk1"
    ).with_options(
        gpu=modal.gpu.A10G(),
        timeout=600,
    )
    model = Model()

    @web_app.get("/health")
    async def health():
        stats = await model.generate.get_current_stats.aio()
        if stats.num_total_runners == 0:
            status_code = fastapi.status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            status_code = fastapi.status.HTTP_200_OK

        response = fastapi.Response(
            content="",
            status_code=status_code,
            media_type="text/plain",
        )
        return response

    @web_app.get("/stats")
    async def stats():
        stats = await model.generate.get_current_stats.aio()
        stats = {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
        }
        return stats

    @web_app.post("/generate")
    async def generate(request: fastapi.Request) -> fastapi.Response:
        content_type = request.headers.get("Content-Type")
        if content_type != "application/json":
            return fastapi.Response(
                content="",
                status_code=fastapi.status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                media_type="text/plain",
            )

        request_data = await request.json()
        generation_request = GenerationRequest(**request_data)
        response = model.generate.remote(**generation_request.dict())
        return GenerationResponse(**response)

    return web_app


## Finally, we can `curl` to the endpoint to engage with the model. Note, that the first request might take a few seconds
## to account for the coldstart, but subsequent calls will be faster.
##
## ```bash
## curl -X "POST" "https://mk1--mk1-chat-endpoint-dev.modal.run/generate" -H 'Content-Type: application/json' -d '{
##   "text": "What is the difference between a llama and an alpaca?",
##   "max_tokens": 512,
##   "eos_token_ids": [1, 2],
##   "temperature": 0.8,
##   "top_k": 50,
##   "top_p": 1.0
## }'
## ```
##
## Given this basic example, there are many ways to extend the functionality of the endpoint. For example, you
## could add authentication, logging, and more complex error handling or load balancing. The only limit is your
## imagination!
