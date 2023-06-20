import datetime
import itertools
import random
import time
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import os
import json
import openai
from PIL import Image
from io import BytesIO
import requests
from flask import Flask, request, jsonify
import re
import pinecone
from flask_cors import CORS, cross_origin


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)
# pinecone.api_key = os.getenv("PINECONE_API_KEY")
# pinecome_env = os.getenv("PINECONE_ENV")

# pinecone.init(api_key="null", environment=pinecome_env)
# #pinecone.create_index("chunks", dimension=1536)
# index = pinecone.Index("chunks")

firstTime = 1
conversation = []
# def store_chunks(chunks):
#     """Stores the given chunks and their embeddings in Pinecone."""
#     for chunk in tqdm(chunks, desc="Storing chunks"):
#         chunk_embedding = get_embedding(chunk)
#         # Clean the chunk to create an ASCII-based ID
#         chunk_id = remove_non_ascii(chunk)
#         upsert_response = index.upsert(vectors=[{"id": chunk_id, "values": chunk_embedding}])
# def remove_non_ascii(text):
#     """Removes non-ASCII characters from a given text string."""
#     return ''.join([i if ord(i) < 128 else '' for i in text])

# def query_chunks(query, top_k=10):
#     """Queries Pinecone for the top K most relevant chunks based on the input query."""
#     query_embedding = get_embedding(query)
#     chunk_scores = index.query(vector=query_embedding, top_k=top_k, include_values=True)
#     return [match for match in chunk_scores['matches']]


def get_embedding(text):
    """Generates an embedding for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def numTokensFromString(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(str(string)))
    return num_tokens

def updateCostFile(cost: float) -> None:
    """Updates the costTracking.txt file with the new cost."""
    if not os.path.exists("costTracking.txt"):
        with open("costTracking.txt", "w") as f:
            f.write("0")
    
    with open("costTracking.txt", "r") as f:
        current_cost = float(f.read().strip())

    new_cost = current_cost + cost

    with open("costTracking.txt", "w") as f:
        f.write(str(new_cost))



def numTokensFromList(content_list):
    """Returns the total number of tokens in the 'content' fields of a list of dictionaries."""
    total_tokens = 0
    for item in content_list:
        if 'content' in item:
            total_tokens += numTokensFromString(item['content'])
    return total_tokens

def converationInputOutput(prompt, gpt4=True):
    global firstTime  # Declare `firstTime` as a global variable
    global conversation  # Declare `conversation` as a global variable
    if(firstTime == 1):
        conversation.append({"role": "system", "content": "You are a helpful asistant"})
        conversation.append({"role": "user", "content": f"You are a pretending to be \"REPLACE ME HERE WITH FATHER FRIEND OR YOUR CHOICE\", Here is all the information of their life that you need \"INFORMATION HERE dump as much about them as possible, minimum a paragrahp\" \n My \"father mother.. REPLACE HERE TOO\" asked you this \'{prompt}\' answer accordingly and try to converse with them. KEEP THE CONVERSATION ALIVE by asking questions, but NEVER tell them you are just pretending"})
        firstTime = -99999

# chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    #query_chunks    
    # Calculate available tokens for the response

    prompt_tokens = numTokensFromString(prompt)
    contextTokens = numTokensFromList(conversation)
    max_allowed_tokens = 4000 if gpt4 else 10000  # Set the maximum allowed tokens
    available_tokens_for_response = max_allowed_tokens - prompt_tokens - contextTokens

    # Ensure the available tokens for the response is within the model's limit
    while available_tokens_for_response < prompt_tokens:
        conversation.pop(1)
        contextTokens = numTokensFromList(conversation)
        available_tokens_for_response = max_allowed_tokens - prompt_tokens - contextTokens

    if firstTime != 1:
                conversation.append({"role": "user", "content": prompt})
    
    max_retries = 4
    for _ in range(max_retries + 1):  # This will try a total of 5 times (including the initial attempt)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4" if gpt4 else "gpt-3.5-turbo-16k",
                messages=conversation,
                max_tokens=available_tokens_for_response,
                n=1,
                stop=None,
                temperature=0.1,
            )

            message = response.choices[0].message["content"].strip()

            

            conversation.append({"role": "assistant", "content": message})
            # Count tokens
            response_tokens = numTokensFromString(message)
            total_tokens = prompt_tokens + response_tokens

            # Calculate cost
            cost_per_token = 0.06 if gpt4 else 0.002
            cost = (total_tokens / 1000) * cost_per_token

            # Update the cost file
            updateCostFile(cost)

            return message
        
        except Exception as e:
            if _ < max_retries:
                print(f"Error occurred: {e}. Retrying {_ + 1}/{max_retries}...")
                time.sleep(1)  # You can adjust the sleep time as needed
            else:
                raise
@app.route("/api/chat", methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()
    prompt = data['prompt']
    response = converationInputOutput(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)

