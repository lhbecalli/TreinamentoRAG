# IMPORTS E INSTALAÇÃO

# Biblioteca de matemática com lista de números,
# serve pra guardar os vetores dos chunks e calcular a similaridade.
import numpy as np

# Biblioteca que carrega os modelos de embedding, (embedding transforma o texto em número.)
from sentence_transformers import SentenceTransformer

import anthropic


chunks = [

    "text: O Desportiva é o maior time de itaguaçu no futebol capixaba",
    "text: O vasco será rebaixado a Série B em 2027",
    "text: A loja de material em Itaguaçu, se chama Retock" 

]

# Embedding
##Baixar o modelo e carregar de embedding
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chunk_vecs = model.encode(chunks, normalize_embeddings=True)

query = "Qual é o nome da loja?"

## Passo 1 - Transformar a pergunta em Vetor, [0] pega esse primeiro (e único) vetor.
q = model.encode([query], normalize_embeddings=True)[0] ## Transforma a pergunta em numeros ( faz a codificação )

## Passo 2 - Dar uma nota para cada chunk
## @ Compara cada pergunta 'q' ao chunks, quanto mais próximo de 1, mais similaridade tem.
score = chunks_vecs @ q ## Da notas a cada chunks

k = 2

top = np.argsort(score)[::-1][k]

for i in top:
    print(f"{score[i]:.2f}") | {chunks[i]}