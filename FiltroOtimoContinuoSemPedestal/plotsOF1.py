import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

ocupacao = 20
n_janelamento = 7


############################################### CARREGAR INFORMAÇÕES PARA MEDIA DA MEDIA ##################################################
caminho_arquivo_dados = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuoSemPedestal/Dados/MediaDaMedia.txt"

# notebook
# caminho_arquivo_dados= "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuoSemPedestal/Dados/MediaDaMedia.txt"

# Função para ler os dados organizados por ocupação
def ler_dados_por_ocupacao(caminho_arquivo):
    dados = {}
    with open(caminho_arquivo, 'r') as arquivo:
        next(arquivo)  # Pula a primeira linha
        for linha in arquivo:
            partes = linha.split()
            janelamento = int(partes[0])
            ocupacao = int(partes[1])
            # Remover a vírgula se houver e então converter para float
            media_da_media = float(partes[2].replace(',', ''))
            desvio_padrao_do_desvio_padrao = float(partes[3].replace(',', ''))
            media_do_desvio_padrao = float(partes[4].replace(',', ''))
            if ocupacao not in dados:
                dados[ocupacao] = {'janelamentos': [],'medias': [], 'desvios': [], 'MediaDesvioPadrao': []}
            dados[ocupacao]['janelamentos'].append(janelamento)
            dados[ocupacao]['medias'].append(media_da_media)
            # Aplicar o módulo ao desvio padrão do desvio padrão
            desvio_padrao_do_desvio_padrao = abs(desvio_padrao_do_desvio_padrao)
            dados[ocupacao]['desvios'].append(desvio_padrao_do_desvio_padrao)
            dados[ocupacao]['MediaDesvioPadrao'].append(media_do_desvio_padrao)
    return dados

# Função para ler os dados organizados por janelamento
def ler_dados_por_janelamento(caminho_arquivo):
    dados = {}
    with open(caminho_arquivo, 'r') as arquivo:
        next(arquivo)  # Pula a primeira linha
        for linha in arquivo:
            partes = linha.split()
            janelamento = int(partes[0])
            ocupacao = int(partes[1])
            media_media = float(partes[2].replace(',', ''))
            desvio_padrao_do_desvio_padrao = float(partes[3].replace(',', ''))
            media_do_desvio_padrao = float(partes[4].replace(',', ''))
            if janelamento not in dados:
                dados[janelamento] = {'ocupacoes': [], 'mediaDaMedia': [],'mediaDesvioPadrao': [], 'desvios': []}
            dados[janelamento]['mediaDaMedia'].append(media_media)
            dados[janelamento]['ocupacoes'].append(ocupacao)
            dados[janelamento]['mediaDesvioPadrao'].append(media_do_desvio_padrao)
            # Aplicar o módulo ao desvio padrão do desvio padrão
            desvio_padrao_do_desvio_padrao = abs(desvio_padrao_do_desvio_padrao)
            dados[janelamento]['desvios'].append(desvio_padrao_do_desvio_padrao)
    return dados


# Função para plotar os gráficos organizados por ocupação
def plotarMediaDaMedia(dados):
    for ocupacao, info in dados.items():
        janelamentos = info['janelamentos']
        medias = info['medias']
        desvios = info['desvios']
        plt.errorbar(janelamentos, medias, yerr=desvios, fmt='-o', label=f'Ocupação {ocupacao}')
    plt.xlabel('Janelamento')
    plt.ylabel('Média da média do erro de estimação (ADC Count)')
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()

# Função para plotar os gráficos
def plotarMediaDesvioPadrao(dados):
    for ocupacao, info in dados.items():
        janelamentos = info['janelamentos']
        mediaDesvioPadrao = info['MediaDesvioPadrao']
        desvios = info['desvios']
        plt.errorbar(janelamentos, mediaDesvioPadrao, yerr=desvios, fmt='-', label=f'Ocupação {ocupacao}')
    plt.xlabel('Janelamento', fontsize=20)
    plt.ylabel('Média do desvio padrão do erro de estimação (ADC Count)',fontsize=16)
    plt.title(r'$Média$ $desvio$ $padrão$ $\times$ $Janelamento$ $(OF1)$', fontsize=21)
    plt.legend(ncol=5, loc=0)
    plt.grid(True)
    plt.show()


# Função para plotar os gráficos organizados por janelamento
def plotDispersao(dados):
    plt.figure(figsize=(8, 6))  # Ajuste os valores conforme necessário para alterar o tamanho da figura
    for janelamento, info in dados.items():
        ocupacoes = info['ocupacoes']
        medias = info['mediaDesvioPadrao']
        desvios = info['desvios']
        plt.errorbar(ocupacoes, medias, yerr=desvios, fmt='-o',linestyle='dashed',  label=f'Janelamento: {janelamento}')
    # Definindo o tamanho da figura
    
    plt.xlabel('Ocupação', fontsize=20)
    plt.ylabel('Dispersão', fontsize=20)
    plt.title(r'$Dispersão$ $\times$ $Ocupação$ $(OF1)$', fontsize=21)
    plt.xticks(range(0, 101, 10), fontsize=18)
    plt.yticks(fontsize=18)  # Define o tamanho da fonte para os rótulos do eixo Y
    plt.legend(ncol=5,loc=0)
    plt.grid(True)
    plt.show()


def plotarMediaJanelamento(dados):
    for janelamento, info in dados.items():
        ocupacoes = info['ocupacoes']
        medias = info['mediaDaMedia']
        desvios = info['desvios']
        if janelamento==19:
            plt.errorbar(ocupacoes, medias, yerr=desvios, fmt='-o', label = "Janelamento: "+str(janelamento), color='purple')
    plt.xlabel('Ocupação', fontsize=18)
    plt.ylabel('Média do erro de estimação (ADC Count)', fontsize=18)
    plt.xticks(range(0, 101, 10), fontsize=18)
    plt.yticks(fontsize=18)  # Define o tamanho da fonte para os rótulos do eixo Y
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()

# Caminho para o arquivo de dados
caminho_arquivo_dados = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuoSemPedestal/Dados/MediaDaMedia.txt"

# notebook
# caminho_arquivo_dados= "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados/MediaDaMedia.txt" 

dadosParaCadaOcupacao = ler_dados_por_ocupacao(caminho_arquivo_dados)
#media da media de cada ocupação para todos os janelamentos
# plotarMediaDaMedia(dadosParaCadaOcupacao)
#media do desvio padrão de cada ocupação para todos os janelamentos
# plotarMediaDesvioPadrao(dadosParaCadaOcupacao)

#Dispersao por ocupação para todos os janelamentos
dadosParaCadaJanelamento = ler_dados_por_janelamento(caminho_arquivo_dados)
plotDispersao(dadosParaCadaJanelamento)
# plotarMediaJanelamento(dadosParaCadaJanelamento)
