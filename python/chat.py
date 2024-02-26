# Gerais
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# Scraping
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from urllib.request import Request, urlopen

# Interface
import gradio as gr

# Assistente
import tiktoken
import openai
import time
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


urls = ["https://www.azapfy.com.br/", "https://www.azapfy.com.br/embarcadora/", "https://www.azapfy.com.br/transportadora/"]
domain = "azapfy.com.br"

url = "https://www.azapfy.com.br/"


def get_text_from_url(url):
    '''
        Essa função recebe uma url e faz o scraping do texto da página
        Retorna o texto da página e salva o texto em um arquivo <url>.txt
    '''
    
    # Analisa a URL e pega o domínio
    local_domain = urlparse(url).netloc
    print(local_domain)

    # Fila para armazenar as urls para fazer o scraping
    fila = deque(urls)
    print(fila)

    # Criar um diretório para armazenar os arquivos de texto
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # Enquanto a fila não estiver vazia, continue fazendo o scraping
    while fila:
        # Pega a próxima URL da fila
        url = fila.pop()
        print("Próxima url",url) # Checa próxima url

        # Salva o texto da url em um arquivo <url>.txt
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding='utf-8') as f:
            driver = Chrome()
            driver.get(url)
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            text = page_soup.get_text()
            f.write(text)

        driver.close()
get_text_from_url(url)

def remove_newlines(serie):
    '''
        Essa função recebe uma série e remove as quebras de linha
    '''
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Criar uma lista para armazenar os arquivos de texto
texts=[]
# Obter todos os arquivos de texto no diretório de texto
for file in os.listdir("text/www." + domain + "/"):
    # Abra o arquivo e leia o texto
    with open("text/www." + domain + "/" + file, "r") as f:
        text = f.read()
        # Omita as primeiras 20 linhas e as últimas 4 linhas e, em seguida, substitua  -, _, e #update com espaços.
        texts.append((file[20:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

#Criar um Dataframe a partir da lista de textos
df = pd.DataFrame(texts, columns = ['page_name', 'text'])

# Defina a coluna de texto para ser o texto bruto com as novas linhas removidas
df['text'] = df.page_name + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

print("Checando número de páginas extraidas e urls informadas \n")
print("Número de páginas",df.shape[0])
print("\nNúmero de urls informadas",len(urls))

# Carregar o tokenizador cl100k_base que foi projetado para funcionar com o modelo ada-002
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize o texto e salve o número de tokens em uma nova coluna
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize a distribuição do número de tokens por linha usando um histograma
df.hist(column='n_tokens')

df.head()

max_tokens = 500

# Função para dividir o texto em partes de um número máximo de tokens
def split_into_many(text, max_tokens = max_tokens):

    # Dividir o texto em frases
    sentences = text.split('. ')

    # Obter o número de tokens para cada sentença
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Percorrer as sentenças e os tokens unidos em uma tupla
    for sentence, token in zip(sentences, n_tokens):

        # Se o número de tokens até o momento mais o número de tokens na frase atual for maior 
        # do que o número máximo de tokens, adicione o bloco à lista de blocos e redefina
        # o bloco e os tokens até o momento
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # Se o número de tokens na frase atual for maior que o número máximo de 
        # tokens, vá para a próxima sentença
        if token > max_tokens:
            continue

        # Caso contrário, adicione a frase ao bloco e adicione o número de tokens ao total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

# Percorrer o dataframe
for row in df.iterrows():

    # Se o texto for None, vá para a próxima linha
    if row[1]['text'] is None:
        continue

    # Se o número de tokens for maior que o número máximo de tokens, divida o texto em partes
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Caso contrário, adicione o texto à lista de textos abreviados
    else:
        shortened.append( row[1]['text'] )


df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.hist(column='n_tokens')


num_tot_tokens = df['n_tokens'].sum()
print("Número total de tokens",num_tot_tokens)

df.head()

i = 0
for text in df['text']:
    i+=1
    
print("Número de trechos de texto com no máximo",max_tokens,"tokens :",i)

print("Custo total de treinamento do embedding: $",num_tot_tokens /1000 * 0.0001)


def read_openai_api_key():
    with open('openai_secret.txt', 'r') as file:
        api_key = file.read().strip()
    return api_key

my_api_key = read_openai_api_key()

openai.api_key = read_openai_api_key()

i = 0
embeddings = []
for text in df['text']:
    time.sleep(2)
    print(i)
    try:
        embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
        print("Fazendo embedding do texto")
        embeddings.append(embedding)
    except openai.error.RateLimitError:
        print("Rate limit error, esperando 20 segundo antes de tentar novamente")
        time.sleep(20)  
        embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
        print("embedding texto depois de esperar 20 segundos")
        embeddings.append(embedding)
    i+=1

df['embeddings'] = embeddings
df.to_csv('processed/embeddings.csv')
df.head()

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()

def create_context(question, df, max_len=1800, size="ada"):
    """
    Cria um contexto para uma pergunta encontrando o contexto mais similar no conjunto de embeddings gerado utilizando o Custom Knowledge.
    """

    # Obter a embeddings para a pergunta que foi feita
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Obter as distâncias a partir dos embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Classifique por distância e adicione o texto ao contexto
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Adicionar o comprimento do texto ao comprimento atual
        cur_len += row['n_tokens'] + 4
        
        # Se o contexto for muito longo, quebre
        if cur_len > max_len:
            break
        
        # Caso contrário, adicione-o ao texto que está sendo retornado
        returns.append(row["text"])

    # Retornar o contexto
    return "\n\n###\n\n".join(returns)

def answer_question(
                    df=df,
                    model="text-davinci-003",
                    question="O que é a bix tecnologia?",
                    max_len=1800,
                    size="ada",
                    debug=False,
                    max_tokens=150,
                    stop_sequence=None):
    """
    Responder a uma pergunta com base no contexto mais semelhante dos textos do dataframe
    """
    context = create_context(
        question,
        df=df,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = openai.Completion.create(
            prompt=f"Responda as perguntas com base no contexto abaixo, e se a pergunta não puder ser respondida diga \"Eu não sei responder isso\"\n\Contexto: {context}\n\n---\n\nPergunta: {question}\nResposta:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
    #     retornar ""