import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ocupacao = 20
n_janelamento = 7

# Caminho do arquivo
caminho_arquivo = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimoContinuo/ErrosEstimacao/ErroEstimacao_" + str(ocupacao) + ".xlsx"

# Ler o arquivo Excel usando pandas
dados = pd.read_excel(caminho_arquivo, dtype={'Matriz_Covariancia': str})

# Inicializar variáveis para armazenar os valores associados ao janelamento desejado
pesos = None
matriz_covariancia = None
erro_estimacao_amplitude = None
media_erro_estimacao = None
desvio_padrao_erro_estimacao = None

# Filtrar o DataFrame para selecionar apenas as linhas com o valor de janelamento desejado
linha_janelamento_desejado = dados[dados['Janelamento'] == n_janelamento]

# Verificar se há linhas correspondentes ao janelamento desejado
if not linha_janelamento_desejado.empty:
    # Iterar sobre as linhas do DataFrame
    for _, linha in linha_janelamento_desejado.iterrows():
        pesos = linha['Pesos']
        matriz_covariancia = np.fromstring(linha['Matriz_Covariancia'], sep=' ')
        erro_estimacao_amplitude = linha['Erro_Estimacao_Amplitude']
        media_erro_estimacao = linha['Media_Erro_Estimacao']
        desvio_padrao_erro_estimacao = linha['Desvio_Padrao_Erro_Estimacao']

    # Imprimir os valores associados ao janelamento desejado
    print("Valores associados ao janelamento", n_janelamento)
    print("Pesos:", pesos)
    print("Matriz de Covariância:", matriz_covariancia)
    print("Erro de Estimação da Amplitude:", len(erro_estimacao_amplitude))
    print("Média do Erro de Estimacao:", media_erro_estimacao)
    print("Desvio Padrão do Erro de Estimacao:", desvio_padrao_erro_estimacao)
    
    # Verificar se a matriz de covariância é válida (não vazia e 2D)
    if matriz_covariancia is not None and matriz_covariancia.ndim == 2:
        # Criar um mapa de calor da matriz de covariância
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_covariancia, annot=True, cmap='coolwarm', fmt=".5f")
        plt.title(f"Matriz de Covariância do Ruído ({ocupacao}% Ocupação)")
        plt.show()
    else:
        print("Matriz de Covariância não é válida.")
else:
    print(f"Nenhuma linha encontrada para o janelamento {n_janelamento}.")
