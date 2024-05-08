import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

ocupacao = 20
n_janelamento = 7

############################################## PLOTS DOS PESOS, COV E DISPERSAO SIMPLES #############################################
# Caminho do arquivo Excel
caminho_arquivo_excel = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/ErroEstimacao/ErroEstimacao_J"+str(n_janelamento)+".xlsx"

# notebook
# caminho_arquivo_excel= "C:\Users\diogo\OneDrive\Área de Trabalho\TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1\FiltroOtimoContinuo/ErroEstimacao/ErroEstimacao_J"+str(n_janelamento)+".xlsx"
# Função para converter uma string em uma matriz NumPy
def string_para_matriz(string):
    # Remover colchetes extras e quebras de linha
    string = re.sub(r'[\[\]\n]', '', string)
    # Dividir a string pelos espaços em branco e converter para float
    valores = [float(valor) for valor in string.split()]
    # Converter a lista de valores em uma matriz NumPy de formato adequado
    matriz = np.array(valores).reshape(-1, len(valores) // 7)
    return matriz

# Função para converter a string do vetor de erro de estimativa de amplitude para uma lista
def string_para_lista(string):
    # Remover colchetes e dividir a string pelos espaços em branco
    valores = string.strip('[]').split()
    # Remover a vírgula extra no final, se houver
    valores = [valor.replace(',', '') for valor in valores]
    # Converter os valores para float e criar a lista
    lista = [float(valor) for valor in valores]
    return lista

# Ler o arquivo Excel
dados_excel = pd.read_excel(caminho_arquivo_excel, sheet_name='Dados')

# Selecionar os dados da linha com a ocupação desejada
dados_ocupacao_desejada = dados_excel.loc[dados_excel['Ocupacao'] == ocupacao]

# Converter os tipos de dados conforme necessário
dados_excel['Ocupacao'] = dados_excel['Ocupacao'].astype(int)
dados_excel['Pesos'] = dados_excel['Pesos'].apply(eval)  # Convertendo para lista
dados_excel['Media_Erro_Estimacao'] = dados_excel['Media_Erro_Estimacao'].astype(float)
dados_excel['Desvio_Padrao_Erro_Estimacao'] = dados_excel['Desvio_Padrao_Erro_Estimacao'].astype(float)
dados_excel['Matriz_Covariancia'] = dados_excel['Matriz_Covariancia'].apply(string_para_matriz)  # Convertendo para matriz

# Inicializar um vetor para armazenar os dados da ocupação desejada
dados_ocupacao_especifica = None

# Verificar se existem dados para a ocupação desejada
if not dados_ocupacao_desejada.empty:
    # Extrair os dados da primeira linha (assume-se que há apenas uma linha para a ocupação desejada)
    dados_ocupacao_especifica = dados_ocupacao_desejada.iloc[0]
else:
    print("Ocupacao nao preenchida")


def PlotMatrizCov(ocupacoes_desejadas):
     # Percorrer as ocupações desejadas
    for ocupacao_desejada in ocupacoes_desejadas:
        # Selecionar os dados da linha com a ocupação desejada
        dados_ocupacao_desejada = dados_excel.loc[dados_excel['Ocupacao'] == ocupacao_desejada]

        # Verificar se há dados para a ocupação desejada
        if not dados_ocupacao_desejada.empty:
            # Extrair a matriz de covariância da primeira linha (assume-se que há apenas uma linha para a ocupação desejada)
            matriz_covariancia = dados_ocupacao_desejada['Matriz_Covariancia'].iloc[0]

            # Verificar se a matriz de covariância não está vazia
            if matriz_covariancia.size != 0:
                # Criar um mapa de calor da matriz de covariância
                plt.figure(figsize=(10, 8))
                sns.heatmap(matriz_covariancia, annot=True, cmap='coolwarm', fmt=".5f")
                # Personalizar os marcadores e rótulos dos eixos
                plt.xticks(np.arange(0.5, matriz_covariancia.shape[1] + 0.5, 1), np.arange(1, matriz_covariancia.shape[1] + 1, 1))
                plt.yticks(np.arange(0.5, matriz_covariancia.shape[0] + 0.5, 1), np.arange(1, matriz_covariancia.shape[0] + 1, 1))
                # Mover os valores do eixo x para o topo da figura
                plt.gca().xaxis.set_ticks_position('top')
                # Adicionar uma legenda explicativa
                plt.title(f"Matriz de Covariância do Ruído ({ocupacao_desejada}% Ocupação)")
                plt.show()

            else:
                print(f"Nao ha dados de matriz de covariancia para a ocupacao {ocupacao_desejada}")
        else:
            print(f"Nao ha dados para a ocupacao {ocupacao_desejada}")



def PlotDispersao(ocupacoes_desejadas):
    ocupacao_plot=[]
    dispersao_plot =[]
    for ocupacao_desejada in ocupacoes_desejadas:
        # Selecionar os dados da linha com a ocupação desejada
        dados_ocupacao_desejada = dados_excel.loc[dados_excel['Ocupacao'] == ocupacao_desejada]
        # Verificar se há dados para a ocupação desejada
        if not dados_ocupacao_desejada.empty:
            ocupacao_plot.append(dados_ocupacao_desejada['Ocupacao'].iloc[0])
            dispersao_plot.append(dados_ocupacao_desejada['Desvio_Padrao_Erro_Estimacao'].iloc[0])
        else:
            print(f"Nao ha dados de ocupacao ou dispersao para {ocupacao_desejada}")
    plt.plot(ocupacao_plot, dispersao_plot)
    plt.show()


def PlotPesos(ocupacoes_desejadas):
    for ocupacao_desejada in ocupacoes_desejadas:
        dados_ocupacao_desejada = dados_excel.loc[dados_excel['Ocupacao'] == ocupacao_desejada]
        plt.figure(figsize=(10, 6)) 
        plt.plot(dados_ocupacao_desejada['Pesos'].iloc[0], label='Ruído Não Branco', color='black')
        plt.xlabel('Amostra') 
        plt.ylabel('Valor do Peso')  
        plt.title(f"Pesos ruído de fundo({ocupacao_desejada}% Ocupação)")
        plt.grid(True)
        plt.legend()  # Adiciona a legenda
        plt.show()  # Mostra o gráfico



ocupacoes_desejadas = [0,10,20, 30,40,50,60,70,80,90,100] #lista de ocupacoes para teste
# PlotDispersao(ocupacoes_desejadas)
# PlotMatrizCov(ocupacoes_desejadas)
# PlotPesos(ocupacoes_desejadas)

###################################################################### PLOTS PARA MEDIA DA MEDIA #####################################################

############################################### CARREGAR INFORMAÇÕES PARA MEDIA DA MEDIA ##################################################
caminho_arquivo_dados = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados/MediaDaMedia.txt"

# notebook
# caminho_arquivo_dados= "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados/MediaDaMedia.txt"

# Função para ler os dados do arquivo txt
def ler_dados(caminho_arquivo):
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
                dados[ocupacao] = {'janelamentos': [], 'medias': [], 'desvios': [], 'MediaDesvioPadrao': []}
            dados[ocupacao]['janelamentos'].append(janelamento)
            dados[ocupacao]['medias'].append(media_da_media)
            # Aplicar o módulo ao desvio padrão do desvio padrão
            desvio_padrao_do_desvio_padrao = abs(desvio_padrao_do_desvio_padrao)
            dados[ocupacao]['desvios'].append(desvio_padrao_do_desvio_padrao)
            dados[ocupacao]['MediaDesvioPadrao'].append(media_do_desvio_padrao)
    return dados

# Função para plotar os gráficos
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
        plt.errorbar(janelamentos, mediaDesvioPadrao, yerr=desvios, fmt='-o', label=f'Ocupação {ocupacao}')
    plt.xlabel('Janelamento')
    plt.ylabel('Média do desvio padrão do erro de estimação (ADC Count)')
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()


# Ler os dados do arquivo
dados = ler_dados(caminho_arquivo_dados)

# Plotar os gráficos
plotarMediaDaMedia(dados)
plotarMediaDesvioPadrao(dados)