import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
from scipy.integrate import simps
import os

########################################## CAMINHO COM OS DADOS DO PULSO DE REFERENCIA #####################################################################
nome_arquivo = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/pulsehi_physics.txt"
janelamento = 7  #Escolha o janelamento que deseja gerar o pulso de referência

# notebook
# nome_arquivo = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/pulsehi_physics.txt"
#etapa para criar o vetor de pulso de referência dinamicamente, de acordo com o tamanho do janelamento pedido
def ler_valores_arquivo(nome_arquivo):
    valores = {} #dicionario onde a chave é o tempo e o valor é a segunda coluna
    with open(nome_arquivo, 'r') as arquivo:
        for linha in arquivo:
            colunas = linha.split()
            if len(colunas) >= 2:  # Verifica se há pelo menos dois valores na linha
                valor_primeira_coluna = float(colunas[0]) #pega o tempo
                valor_segunda_coluna = float(colunas[1]) #pega o valor associado ao tempo
                valores[valor_primeira_coluna] = valor_segunda_coluna
    return valores

def gerar_vetor_g(nome_arquivo, janelamento): #gerar o pulso de referencia dinamicamente
    valores_arquivo = ler_valores_arquivo(nome_arquivo)
    valor_central = valores_arquivo.get(0, 0)  # Valor associado a 0, ou 0 se não encontrado
    g = [None] * janelamento #vetor de pulsos do tamanho do janelamento
    indice_central = janelamento // 2 #metade do vetor, onde entra o valor para o tempo 0
    for i in range(indice_central):
        valor_esquerda = -25 * (i + 1) #anda de 25 em 25 para a esquerda
        valor_direita = 25 * (i + 1) #anda de 25 em 25 para a direita
        g[indice_central - (i + 1)] = valores_arquivo.get(valor_esquerda, 0) #preenche os valores a esquerda do 0, caso hajam valores associados, caso não, preenche com 0
        g[indice_central + (i + 1)] = valores_arquivo.get(valor_direita, 0) #preenche os valores a direita do 0, caso hajam valores associados, caso não, preenche com 0
    g[indice_central] = valor_central # a posicao do meio é o valor associado ao tempo 0
    return g

g = gerar_vetor_g(nome_arquivo, janelamento)
print("g", g)

# identificar os valores de tempo utilizados para plotar o g pelo tempo
primeira_coluna = []
with open(nome_arquivo, 'r') as arquivo:
    for linha in arquivo:
        colunas = linha.split()
        if len(colunas) >= 2:
            primeira_coluna.append(float(colunas[0]))

# ajustar o valor do tempo de acordo com o janelamento
indice_central_primeira_coluna = len(primeira_coluna) // 2
tempo = primeira_coluna[indice_central_primeira_coluna - janelamento//2:indice_central_primeira_coluna + janelamento//2 + 1] #passa onde a lista deve começar e onde deve terminar


########################################################### CALCULAR DERIVADA ##################################################

if janelamento==7:
    derivada_normalizada = [0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336]
elif janelamento==9: 
    derivada_normalizada = [0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613]
elif janelamento==11:
    derivada_normalizada = [0,0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613,-0.00215426]
elif janelamento==13:
    derivada_normalizada = [0,0,0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613,-0.00215426,0]
elif janelamento==15:
    derivada_normalizada = [0,0,0,0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613,-0.00215426,0,0]
elif janelamento==17:
    derivada_normalizada = [0,0,0,0,0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613,-0.00215426,0,0,0]
elif janelamento==19:
    derivada_normalizada = [0,0,0,0,0,0,0.00005472,0.00367031,0.0310805,0.00000016,-0.0243455,-0.00800683,-0.00243336,-0.00053613,-0.00215426,0,0,0,0]
    
# Caminho do arquivo onde os dados do pulso e sua referência serão salvos
nome_arquivo_saida = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/valores_g_derivada.txt"

# notebook
# nome_arquivo_saida = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/valores_g_derivada.txt"
titulos= ["Janelamento", "g", "derivada_g"]
# Verificar se o arquivo existe, e se não, criar
if not os.path.exists(nome_arquivo_saida):
    with open(nome_arquivo_saida, 'w') as arquivo:
        arquivo.write(' '.join(titulos) + '\n')

# Verificar se já existe uma linha com o número do janelamento
with open(nome_arquivo_saida, 'r') as arquivo_leitura:
    linhas = arquivo_leitura.readlines()

linha_existente = None
for indice, linha in enumerate(linhas):
    if linha.startswith(str(janelamento) + ' '):
        linha_existente = indice
        break

# Se a linha existir, sobrescrever com os novos valores
if linha_existente is not None:
    linhas[linha_existente] = f"{janelamento} {g} {derivada_normalizada}\n"
    # Reescrever todo o arquivo com as alterações
    with open(nome_arquivo_saida, 'w') as arquivo_escrita:
        arquivo_escrita.writelines(linhas)
    print(f"Dados atualizados para janelamento {janelamento}")
else:
    # Se a linha não existir, adicionar como uma nova linha no final do arquivo
    with open(nome_arquivo_saida, 'a') as arquivo:
        arquivo.write(f"{janelamento} {g} {derivada_normalizada}\n")
    print(f"Dados adicionados para janelamento {janelamento}")

# Plotar
# plt.plot(tempo, g, color="C0", label=r'$g$')
# plt.plot(tempo, derivada_normalizada, color='C1', label=r'$\dot g$')
# plt.xlabel('Tempo(ns)')
# plt.ylabel(r'$g$ e $\dot g$')
# plt.title(r'Pulso de referência $\times$ Tempo')
# plt.legend(loc=0)
# plt.show()