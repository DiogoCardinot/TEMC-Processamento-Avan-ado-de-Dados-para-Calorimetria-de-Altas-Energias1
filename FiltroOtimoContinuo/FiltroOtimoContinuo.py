import numpy as np
import os
import pandas as pd

# Número do janelamento desejado
n_janelamento = 7
pedestal = 30
ocupacao = 20

############################################### CARREGAR INFORMAÇÕES DO PULSO DE REFERÊNCIA E DERIVADA ##################################################
nome_arquivo_saida = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/valores_g_derivada.txt"

# notebook
# nome_arquivo_saida = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/valores_g_derivada.txt"
# Inicializar variáveis para armazenar os dados de g e sua derivada
g = None
derivada_g = None

# Abrir o arquivo e procurar pelo janelamento desejado
with open(nome_arquivo_saida, 'r') as arquivo_leitura:
    for linha in arquivo_leitura:
        # Dividir a linha em partes separadas por espaço
        partes = linha.split()
        # Verificar se há elementos suficientes na lista partes
        if len(partes) >= n_janelamento+1:
            # Se a primeira parte for igual ao número do janelamento desejado
            if int(partes[0]) == n_janelamento:
                # Converter as partes relevantes para os vetores desejados
                g = np.array(eval(' '.join(partes[1:n_janelamento+1])))
                derivada_g = np.array(eval(' '.join(partes[n_janelamento+1:])))
                break  # Parar de percorrer o arquivo após encontrar os dados

# Verificar se os dados foram encontrados e carregados corretamente
if g is not None and derivada_g is not None:
    print("Dados de g carregados:\n", g)
    print("Dados de derivada de g carregados:\n", derivada_g)
else:
    print("Dados para o janelamento especificado não foram encontrados ou estão incompletos.")


#################################### LER AS AMOSTRAS E AS AMPLITUDES ASSOCIADAS DE ACORDO COM O JANELAMENTO ###################################################################

nome_arquivo_amostras = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados_Ocupacoes/OC_"+str(ocupacao)+".txt"

# notebook
# nome_arquivo_amostras = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/Dados_Ocupacoes/OC_"+str(ocupacao)+".txt"
dados_Ocupacao = np.genfromtxt(nome_arquivo_amostras, delimiter=",", skip_header=1)

# Calcular o número de linhas para a matriz
num_linhas = len(dados_Ocupacao) - (n_janelamento-1)
# print("Número de linhas da matriz", num_linhas)

# Inicializar a matriz para armazenar os blocos de amostras, com num_linhas linhas e n_janelamento colunas
matriz_Ocupacao_Amostras = np.zeros((num_linhas, n_janelamento))

# Preencher a matriz com os blocos de amostras
for i in range(num_linhas):
    inicio = i #começa no ponto atual
    fim = i + n_janelamento #vai até o ponto atual + n_janelamento
    matriz_Ocupacao_Amostras[i] = dados_Ocupacao[inicio:fim, 1]  # salva a linha na matriz, indo do começo até o final para a coluna 1 ("sample")

# print("Matriz de ocupação:\n", matriz_Ocupacao_Amostras)

# Calcular o índice do valor central em cada linha
indice_central = n_janelamento // 2

# Inicializar um array para armazenar as amplitudes associadas
amplitude_Real = np.zeros(num_linhas)

# Preencher o array de amplitudes associadas
for i in range(num_linhas):
    amplitude_Real[i] = dados_Ocupacao[i + indice_central, 2] #salva a amplitude como sendo os dados na posição atual mais o indice central da amostra em questão e pega a coluna 2 ("Amplitude")

# print("Amplitudes real do sinal:\n" , amplitude_Real)


########################################################### COMECAR O PROCESSO DO FILTRO ÓTIMO DE FATO #####################################################################

caminho_arquivo_ruido= np.loadtxt('C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimo/Dados Estimação/RuidoOcupacao_'+str(ocupacao)+'.txt')

# notebook
# caminho_arquivo_ruido = np.loadtxt('C:\Users\diogo\OneDrive\Área de Trabalho\TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimo/Dados Estimação/RuidoOcupacao_'+str(ocupacao)+'.txt')

# Calcular a matriz de covariância do ruido
cov_matrix_ruido = np.cov(caminho_arquivo_ruido, rowvar=False)

# print("Matriz Covariancia do Ruido\n", cov_matrix_ruido)

A = np.zeros((n_janelamento+3, n_janelamento+3)) #matriz A do sistema para encontrar os pesos

# preenche as primeiras n_janelamento casas(tanto linhas como colunas) da matriz A com a matriz de covariancia do ruido
for i in range(n_janelamento):
    for j in range(n_janelamento):
        A[i][j] = cov_matrix_ruido[i][j]


for i in range(n_janelamento):
    A[i][n_janelamento] = -g[i] 
    A[i][n_janelamento+1] = -derivada_g[i]
    A[i][n_janelamento+2] = -1
    A[n_janelamento][i] = g[i]
    A[n_janelamento+1][i] = derivada_g[i]
    A[n_janelamento+2][i] = 1

# print("Matriz A do sistema para encontrar os pesos\n", A)

B = np.zeros((n_janelamento+3, 1))
B[n_janelamento]=1

#funcao para verificar se o problema tem solucao ou nao
def VerificaSolucao(matrizA, matrizB):
        matrizAAmpliada = np.hstack((matrizA, matrizB))
        # print(matrizAAmpliada)
        postoA = np.linalg.matrix_rank(matrizA)
        postoAAmpliada = np.linalg.matrix_rank(matrizAAmpliada)
        n= matrizA.shape[1] #pega a quantidade de variáveis da matriz A
        if(postoA != postoAAmpliada):
                return "Sistema nao possui solucao!"
        else:
           if(postoA == n):
                solucao = np.linalg.solve(matrizA, matrizB)  
                return solucao
           if(postoA < n):
                 return "Sistema com multiplas solucoes"
           
#solucao do sistema linear matricial
solucaoSistema = VerificaSolucao(A,B)

#vetor de pesos
w=[]

if type(solucaoSistema) != str:  
    for i in range(len(solucaoSistema)):
        #adiciona dinamicamente ao vetor de pesos
        if(i<=n_janelamento-1):
            w.append(solucaoSistema[i][0])
else:
    print(solucaoSistema)

# print("Vetor de pesos:\n", w)
# print("Soma vetor pesos: ", sum(w))

multiplicacao =0
amplitude_Estimada = []


for i in range(len(matriz_Ocupacao_Amostras)): #linhas
    soma = 0 #cada linha a soma vai ser atualizada
    for j in range(n_janelamento): #colunas
        multiplicacao = (matriz_Ocupacao_Amostras[i][j] - pedestal)*w[j]
        soma+=multiplicacao
    amplitude_Estimada.append(soma)

# print("Amplitude estimada: \n", amplitude_Estimada)
# print("Tamanho amplitude estimada:", len(amplitude_Estimada))
# print("Tamanho amplitude real:", len(amplitude_Real))

erroEstimacaoAmplitude = []

for i in range(len(amplitude_Real)):
    erroEstimacaoAmplitude.append(amplitude_Real[i]-amplitude_Estimada[i])

# print("Erro estimacao da amplitude:\n", erroEstimacaoAmplitude)
# print("Tamanho da matriz de erro de estimacao: ", len(erroEstimacaoAmplitude))

mediaErroEstimacao = np.mean(erroEstimacaoAmplitude)
desvioPadraoErroEstimacao = np.std(erroEstimacaoAmplitude)
# print("Media do erro de estimacao:", mediaErroEstimacao)
# print("Desvio padrao do erro de estimacao:", desvioPadraoErroEstimacao)



############################################# SALVAR AS INFORMAÇÕES RELEVANTES EM UM EXCEL #############################################################

# Definir os títulos das colunas
titulos = ['Ocupacao', 'Pesos', 'Matriz_Covariancia', 'Erro_Estimacao_Amplitude', 'Media_Erro_Estimacao', 'Desvio_Padrao_Erro_Estimacao']

# Criar um DataFrame com os seus dados
dados_dict = {
    'Ocupacao': [ocupacao],
    'Pesos': [w],
    'Matriz_Covariancia': [cov_matrix_ruido],
    'Erro_Estimacao_Amplitude': [erroEstimacaoAmplitude],
    'Media_Erro_Estimacao': [mediaErroEstimacao],
    'Desvio_Padrao_Erro_Estimacao': [desvioPadraoErroEstimacao]
}
dados_df = pd.DataFrame(dados_dict)

# Caminho do arquivo Excel de saída
caminho_arquivo_excel = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/ErroEstimacao/ErroEstimacao_J"+str(n_janelamento)+".xlsx"

# notebook
# caminho_arquivo_excel = "C:\Users\diogo\OneDrive\Área de Trabalho\TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/ErrosEstimacao/ErroEstimacaoJ_"+str(n_janelamento)+".xlsx"

# Verificar se o arquivo Excel já existe
arquivo_existe = os.path.exists(caminho_arquivo_excel)

# Escolher o modo do ExcelWriter com base na existência do arquivo
modo = 'a' if arquivo_existe else 'w'

# Definir a opção if_sheet_exists apenas para o modo de escrita "append"
if_sheet_exists = 'overlay' if modo == 'a' else None

with pd.ExcelWriter(caminho_arquivo_excel, mode=modo, engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
    if arquivo_existe:
        # Verificar se o arquivo Excel está vazio
        if os.path.getsize(caminho_arquivo_excel) == 0:
            # Se o arquivo estiver vazio, inserir os dados diretamente
            dados_df.to_excel(writer, index=False, header=True, sheet_name='Dados')
            print("Dados salvos no arquivo Excel.")
        else:
            # Se o arquivo não estiver vazio, ler os dados existentes
            dados_existente = pd.read_excel(caminho_arquivo_excel, sheet_name='Dados')
            # Verificar se a ocupação já existe nos dados existentes
            linha_existente_index = dados_existente[dados_existente['Ocupacao'] == ocupacao].index
            if not linha_existente_index.empty:
                # Se a ocupação já existe, substituir os valores na linha existente pelos novos dados
                dados_existente.loc[linha_existente_index] = dados_df.values
            else:
                # Se a ocupação não existe, adicionar uma nova linha com os novos dados
                novo_dados = pd.concat([dados_existente, dados_df], ignore_index=True)
                dados_existente = novo_dados
            # Sobrescrever o arquivo Excel com os novos dados
            dados_existente.to_excel(writer, index=False, header=True, sheet_name='Dados')
            print("Dados salvos no arquivo Excel.")
    else:
        # Se o arquivo não existir, salve os dados diretamente
        dados_df.to_excel(writer, index=False, header=True, sheet_name='Dados')
        print("Dados salvos no arquivo Excel.")