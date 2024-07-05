import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Função para ler os dados organizados por janelamento
def ler_dados_por_janelamentoOF1(caminho_arquivoOF1):
    dados = {}
    with open(caminho_arquivoOF1, 'r') as arquivo:
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


# Função para ler os dados organizados por janelamento
def ler_dados_por_janelamentoOF2(caminho_arquivoOF2):
    dados = {}
    with open(caminho_arquivoOF2, 'r') as arquivo:
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


def plotComparacaoOF1OF2():
    caminho_OF1 = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuoSemPedestal/Dados/MediaDaMedia.txt"
    dispersaoOF1 = ler_dados_por_janelamentoOF1(caminho_OF1)
    caminho_OF2 = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados/MediaDaMedia.txt"
    dispersaoOF2 = ler_dados_por_janelamentoOF2(caminho_OF2)
    for janelamento, info in dispersaoOF1.items():
        ocupacoesOF1 = info['ocupacoes']
        mediasOF1 = info['mediaDesvioPadrao']
        desviosOF1 = info['desvios']
        if janelamento==19:
            plt.errorbar(ocupacoesOF1, mediasOF1, yerr=desviosOF1, fmt='-o',linestyle='dashed',  label=f'OF1')

    for janelamento, inf in dispersaoOF2.items():
        ocupacoesOF2 = inf['ocupacoes']
        mediasOF2 = inf['mediaDesvioPadrao']
        desviosOF2 = inf['desvios']
        if janelamento==19:
            plt.errorbar(ocupacoesOF2, mediasOF2, yerr=desviosOF2, fmt='-o',linestyle='dashed',  label=f'OF2')
    # Definindo o tamanho da figura
    
    plt.xlabel('Ocupação', fontsize=16)
    plt.ylabel('Dispersão', fontsize=16)
    plt.title(r'$Dispersão$ $OF1$ $\times$ $OF2$ $(Janelamento$ $19)$', fontsize=16)
    plt.xticks(range(0, 101, 10), fontsize=16)
    plt.yticks(fontsize=18)  # Define o tamanho da fonte para os rótulos do eixo Y
    plt.legend(ncol=5, loc=0)
    plt.grid(True)
    plt.show()


plotComparacaoOF1OF2()