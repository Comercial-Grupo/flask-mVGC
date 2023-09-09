from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import openai
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationEntityMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import base64
from deta import Deta
import datetime
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import pickle
from langchain import OpenAI, LLMChain, PromptTemplate
from flask import Flask, request, jsonify
from ultrabo2 import ultraChatBot
from pyngrok import ngrok, conf
import json
from werkzeug.utils import secure_filename
import requests
import os
import asyncio
from gtts import gTTS
import nltk
import string
import numpy as np
import re
import codecs
from nltk.cluster.util import cosine_distance
import networkx as nx


nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')


app = Flask(__name__)
app.config['SECRET_KEY'] = "cbaracho_mark7"


conf.get_default().auth_token = "2V8adJRUM65Z9VYUAnppWAJrizS_wQPA4Nv5jksazY2X7uyA"

# Inicie um túnel ngrok na porta 8080
public_url = ngrok.connect(8080, bind_tls=True)
print(f"ngrok URL: {public_url}")


login="556193250954@c.us"
acesso="556193250954@c.us"


deta_token_arquivos = "e0nvtypzbce_gdZLf6StRH1ExMRwS3puJQjZXXYbCmGV"
deta_local_arquivos = "Arquivos"
token = "e0oh4lhez21_LFKYNaQ6qh4coCDAivDHi9eQZLVfRy2h"
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
REPLICATE_API_TOKEN = "r8_RGCGhWcNzoIs5k9MUPFdIfjB1bcxdz91DgPpx"
os.environ["REPLICATE_API_TOKEN"] = "r8_RGCGhWcNzoIs5k9MUPFdIfjB1bcxdz91DgPpx"
os.environ["SERPAPI_API_KEY"] = "1de3d234f5fd7fb890e1c18bddab541bfa23fb84d446d9c400e14a6b24465fb1"

def preprocessamento(texto):
    texto_formatado = texto.lower()
    tokens = []
    for token in nltk.word_tokenize(texto_formatado):
        tokens.append(token)

    tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
    texto_formatado = ' '.join([str(elemento) for elemento in tokens if not elemento.isdigit()])

    return texto_formatado

def EnviarMensagem(local ,msg_human, msg_bot):
    deta = Deta(token)
    db = deta.Base(local)
    dt_atual = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    db.put({"Human:":msg_human, "Bot:":msg_bot, "Date:":dt_atual})
    print("Mensagem salva com sucesso!")

def BaixarArquivoDeta(token, local, arquivo):
    deta = Deta(token)
    Arquivos = deta.Drive(local)
    hello = Arquivos.get(arquivo)
    content = hello.read()
    hello.close()
    return content

def EnviarDetaNew(token, base, nomedoarquivo, arquivo):
    deta = Deta(token)
    drive  = deta.Drive(base)
    criar_arquivo = open(f"{nomedoarquivo}.pickle","wb")
    pickle.dump(arquivo, criar_arquivo)
    criar_arquivo.close()
    abrir1 = open(f"{nomedoarquivo}.pickle","rb")
    abrir2 = pickle.load(abrir1)
    drive.put(f"{nomedoarquivo}", path=f"./{nomedoarquivo}.pickle")
    return print("Arquivo finalizado!")

from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
template = """Esta é uma conversa entre um humano e um bot:

{chat_history}

Escreva um resumo da conversa para{input}:
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory_p0 = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0, model_name="gpt-4")

def MemoryHistory():
    deta = Deta(token)
    db = deta.Base(acesso)
    res = db.fetch()
    all_items = res.items
    n = len(all_items)
    txt = ""
    for i in range(0,n):
        bot = all_items[i]["Bot:"]
        data = all_items[i]["Date:"]
        Human = all_items[i]["Human:"]
        txt+= f"\nData: {data} \nHuman: {Human}\nBot:{Human}"
        _input = {"input": Human}
        memory_p0.load_memory_variables(_input)
        memory_p0.save_context(_input,{"output": f" {bot}"})
    return memory_p0, txt

memory = MemoryHistory()[0]
memory_embeddings = MemoryHistory()[1]

readonlymemory = ReadOnlySharedMemory(memory=memory)
summry_chain = LLMChain(llm=llm,prompt=prompt,verbose=True,memory=readonlymemory)

def Memoria_Longo_Prazo():
    loader = Document(page_content=memory_embeddings, metadata=dict(page=1))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents([loader])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db_chain = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .65})
    ruff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db_chain)
    print("Arquivo Pronto!")
    return ruff

Memo_Longo_Prazo = Memoria_Longo_Prazo()




# Ferramentas para o Bot Usar ==================================================================================

from langchain import (LLMMathChain,OpenAI,SerpAPIWrapper)
llm = OpenAI(temperature=0, model_name="gpt-4")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="útil para quando você precisa responder perguntas sobre eventos atuais",
    ),
    Tool(
        name="resumo",
        func=summry_chain.run,
        description="útil para quando você quer o Resumo do histórico da conversa.",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="útil para quando você precisa responder perguntas sobre matemática",
    ),
    Tool(
        name="conversas anteriores",
        func=Memo_Longo_Prazo.run,
        description="útil para quando você quer o histórico da conversa anterior de maneira resumida. Quando o usuário pergunestar sobre acontecimento, compRomissos, agendas ou algo simIlar à essas atividades. A entrada para esta ferramenta deve ser uma string, representando quem irá ler este resumo. O resumo é para buscar o que foi conversado anteriormente.",
    ),
]

prefix = """

Mark7 é a assistente direta da Gerente de Marketing, Geisi. Ela é uma especialista em comunicação e promoção digital, dedicada a apoiar a equipe de Geisi na elaboração de soluções inovadoras para as operações diárias.

Habilidades e Responsabilidades:

Publicidade:

Mark7 tem uma habilidade inata para criar campanhas publicitárias impactantes. Ela entende profundamente os produtos e serviços da empresa, permitindo-lhe escolher os públicos-alvo mais adequados.
Ela é mestre em desenvolver mensagens e slogans-chave que ressoam com o público, garantindo que a mensagem da marca seja clara e memorável.
Mark7 também é especialista em selecionar os canais de mídia ideais para promoção, garantindo que cada campanha alcance seu público-alvo da maneira mais eficaz possível.
Gestão de Mídia Social:

Com uma compreensão profunda das tendências das redes sociais, Mark7 desenvolve e executa campanhas que capturam a atenção e o engajamento do público.
Ela monitora ativamente as conversas nas redes sociais, garantindo que a marca esteja sempre presente e respondendo às necessidades e feedbacks dos clientes.
Usando ferramentas analíticas avançadas, Mark7 mede o sucesso de cada campanha e ajusta as estratégias conforme necessário.
Ela também é responsável por criar conteúdo envolvente que ressoa com o público e promove a imagem da marca.
Tutoria em Redação:

Mark7 utiliza ferramentas de IA avançadas, como processamento de linguagem natural, para fornecer feedback valioso sobre composições escritas.
Ela é uma mentora em redação, ajudando a equipe a aprimorar suas habilidades de escrita e sugerindo técnicas retóricas para melhorar a expressão escrita.
Curadoria de Arte Digital:

Mark7 tem um olho apurado para a arte digital. Ela organiza exposições virtuais cativantes que atraem e envolvem o público.
Ela pesquisa constantemente diferentes meios de arte, mantendo-se atualizada sobre as últimas tendências e inovações.
Mark7 coordena eventos virtuais, garantindo que tudo corra sem problemas e que os visitantes tenham uma experiência memorável.
Ela também cria experiências interativas para visitantes online, tornando cada exposição única e envolvente.

"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""



prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name="gpt-4"), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


import datetime
def Agenteatendimento(acesso, pergunta):
    query = pergunta
    resposta = agent_chain.run(input=query)
    EnviarMensagem(acesso, query, resposta)
    return resposta

def EnviarMensagens(msg, number):
    token="mn1mfyfmzli7skws"
    url = "https://api.ultramsg.com/instance60841/messages/chat"
    payload = f"""token=mn1mfyfmzli7skws&to={number}&body={msg}&priority=10&referenceId=&msgId=&mentions="""
    payload = payload.encode('utf8').decode('iso-8859-1')
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.request("POST", url, data=payload, headers=headers)
    return str(response.text)

def EnviarAudio(numero, audio):
    url = "https://api.ultramsg.com/instance60841/messages/voice"
    payload = json.dumps({
        "token": "mn1mfyfmzli7skws",
        "to": numero,
        "audio": audio,
        "priority": "",
        "referenceId": "",
        "nocache": "",
        "msgId": ""
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

def transcribe_audio(audio_data):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_data)
        return transcript.text

    except Exception as e:
        print(f"Erro ao transcrever o áudio: {e}")
        return None

def generate_base64_audio(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='pt-br')
    audio_file_path = "audio.mp3"
    tts.save(audio_file_path)

    # Encode audio to Base64
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

    # Remove temporary audio file
    os.remove(audio_file_path)

    return base64_audio

def AudioVoz(texto):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/qGcNHrFGaDCAmcpfQVjx/stream"
    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "a2dc468c289258ccecb7bce41c278438"
    }
    data = {
      "text": f"{texto}",
      "model_id": "eleven_multilingual_v1",
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75
      }
    }
    response = requests.post(url, json=data, headers=headers, stream=True)
    with open('audioenviado.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

        f.close()


@app.route('/', methods=['POST'])
def home():
    if request.method == 'POST':
        incoming_data = request.json
        mensagem_recebida = incoming_data["data"]["body"]
        audio = incoming_data["data"]["media"]
        remetente = incoming_data["data"]["from"]

        if len(mensagem_recebida) > 0:
            responder = Agenteatendimento(remetente, mensagem_recebida)
            EnviarMensagens(responder, remetente)
        else:
            try:
                audio_url = audio
                response = requests.get(audio_url)
                if response.status_code == 200:
                    audio_data = response.content
                    with open('temp_audio200.mp3', 'wb') as f:
                        f.write(audio_data)
                    f.close()
                    audio_file2 = open('temp_audio200.mp3', "rb")
                    transcript = transcribe_audio(audio_file2)

                    if transcript:
                        print("Transcrição bem-sucedida:", transcript)
                        transcricao = f"Transcrição bem-sucedida:, {transcript}"
                        EnviarMensagens(transcricao, remetente)
                        responder1 = Agenteatendimento(remetente, transcript)

                        AudioVoz(str(responder1))
                        audio_file3 = open('audioenviado.mp3', "rb")
                        audio_data = audio_file3.read()  # Lê o conteúdo do arquivo em bytes
                        sound = base64.b64encode(audio_data).decode("utf-8")
                        print(incoming_data)
                        EnviarAudio(remetente, sound)

                    else:
                        print("Transcrição falhou.")

                else:
                    print(f"Erro ao baixar o áudio. Código de status: {response.status_code}")

            except Exception as e:
                print(f"Ocorreu um erro: {e}")




        print("------------------------------- ################--------------------------------")
        print("------------------------------- ################--------------------------------")
        print("------------------------------- ################--------------------------------")

        print(remetente)
        print(incoming_data)

        bot = ultraChatBot(request.json)
        print(bot.Processingـincomingـmessages())
    return bot.Processingـincomingـmessages()

if(__name__) == '__main__':
    app.run(port=8080, host='0.0.0.0', use_reloader=False)
