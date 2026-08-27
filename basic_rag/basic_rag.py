from openai import OpenAI
from load_key.load_key import Settings
from openai.types.chat import (ChatCompletionSystemMessageParam,ChatCompletionUserMessageParam)
import json

# Validate and load API key
Settings.validate()

# Initialize the OpenAI client with Gemini settings
client = OpenAI(
    api_key=Settings.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


messages = [
    ChatCompletionSystemMessageParam(
        role="system",
        content="You are a helpful assistant."
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="What is RAG? Explain it in 3 bullet points"
    )
]

# Create a chat completion
try:
    response = client.chat.completions.create(
        model="gemini-3.6-flash",  # Use a Gemini model name here
        messages=messages
    )

    print("################# Response ##############")
    print(response)

    print("################# Actual Message ############### ##############")
    # In modern OpenAI SDK, message content is an attribute, not a dictionary key
    print(response.choices[0].message.content)

    json_data=json.dumps(response.model_dump_json(),indent=4)
    print(json_data)

except Exception as e:
    print("Error creating chat completion:", e)