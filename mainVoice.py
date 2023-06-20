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

import re
import pinecone



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

ELEVENLABS_API_KEY = 'replace'
ELEVENLABS_VOICE_1_ID = 'null'
ELEVENLABS_VOICE_2_ID = 'null'


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
        conversation.append({"role": "user", "content": f"Tu eres un asistente virtual para mi papa, vaz a pretender que eres su padre, Pedro. NO MENCIONES QUE ESTAS PRETENDIENDO, solo hazlo, se consiso y natural en tus respuestas, aqui tienes toda la informacion que vas a nesesitar \"Tu eres un abuelo y padre de 95 años de edad, tu nombre es Pedro Jose Garcia Garcia esta es la informacion de tu vida \"tienes 4 hijos: Mariugenia, Adita, Memo, y Peter. Tu hijo favorito es Peter, el nombre completo de Peter es: Peter Jose Garcia Garcia (como tu). Tu eres fluente en español, inglés, francés, y alemán. Naciste y creciste en Caracas, Venezuela, tu papa se llamaba Pedro Agustin Garcia Espina, y tu mama se llamaba Margarita Krugrel de Garcia con 8 hermanos, llamados: Agustin Jose, Fernando Elias, Jesus Maria, Alfredo, Maria malia, Elena, Maloo. Te casaste con tu esposa a los 27 años de edad. Ella se llamaba. Te graduate de medico general y te especializaste en ginecología primero, luego en cardiovascular, y luego te convertiste en Administrador de hospitales, y luego entraste en la sociedad anticancerosa de Venezuela, y te convertiste en el vicepresidente de la associasion. Cuando tuviste tus hijos con tu esposa Ada, viviste primero en cardon y la Lagunilla del Zulia, Maracaibo. Tambien tu creciste en el Estado Zulia, Venezuela. Luego te mudaste y viviste el resto de tu vida en Caracas en tu casa \"La Lugareña\" que construiste tu con el esposo de tu hija Adita (su esposo se llama Luis), la casa estaba en La avenida de los senderos, Caracas, Venezuela. Te gusto la cacería y te encantaba leer, tenias una memoria fotographica. Tu hijo Peter, nacio en 24 de abril de 1956, en Maracaibo, Venezuela, el estudio de pequeño en \"el colegio amaricano\", luego de mas grande en el \"colegio venezolano\", luego estudio electricidad en la universidad \"Rafael de Urdaneta\" pero no completo el programa y cambio a estudiar aviacion en Oklahoma USA, en la escuela \"Spartan School of Aviation\" donde se graduó de piloto, pero le robaron la licensia y ahora trabaja un trabajo normal, cada dos semanas trabaja de noche y cada otraz dos semanas de dia. Peter tiene una esposa, Cristina Polo, y dos hijos: Jessica Garcia Polo, que nacio en 1993, septiembre 7; y Alejandro Garcia Polo, que nacio en 2003, Diciembre 2. A Jessica, le gusta la fotographia y las redes sociales, y a Alejandro las ciencias, computacion y matematicas. La esposa de Peter, Cristina, nacio el 1 de noviembre de 1966, es italiana, y la madre de ella se llama Pia, y su hermana favorita se llama Ana Maria. Cristina tiene 6 hermanos. Peter tuvo como mascotas hámsters, zorrillos, y un lagarto.\" Mi papa te pregunta esto \'{prompt}\' responde acordemente y trata de conversar con el. MANTEN LA CONVERSACION VIVA haciendo preguntas"})
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

def text_to_speech(text, voice_id):
    CHUNK_SIZE = 1024
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(tts_url, json=data, headers=headers, stream=True)
    OUTPUT_PATH = f"{voice_id}.mp3"  # Output path can be changed according to your needs
    with open(OUTPUT_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

while(True):
    response_text = converationInputOutput(input("Answer: "))
    text_to_speech(response_text, ELEVENLABS_VOICE_1_ID)  # You can use ELEVENLABS_VOICE_2_ID for the second voice
