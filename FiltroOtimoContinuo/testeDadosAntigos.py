import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Número do janelamento desejado
n_janelamento = 7
pedestal = 0
ocupacao = 100

############################################### CARREGAR INFORMAÇÕES DO PULSO DE REFERÊNCIA E SUA DERIVADA ##################################################
# nome_arquivo_saida = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/valores_g_derivada.txt"

# notebook
nome_arquivo_saida = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/valores_g_derivada.txt"
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
    # print("Dados de g carregados:\n", g)
    # print("Dados de derivada de g carregados:\n", derivada_g)
    print("\n\n")
else:
    print("Dados para o janelamento especificado não foram encontrados ou estão incompletos.")


#################################### LER AS AMOSTRAS E AS AMPLITUDES ASSOCIADAS DE ACORDO COM O JANELAMENTO ###################################################################

#////////////////////////////////////////// LEITURA DOS ARQUIVOS ///////////////////////////////////////////////
# Ler os dados do arquivo de ruido
# data = np.loadtxt('C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimo/Dados Estimação/RuidoOcupacao_50.txt')

#ler os dados de ruido notebook
data = np.loadtxt('C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimo/Dados Estimação/RuidoOcupacao_0.txt')


#///////////////////////////////////// PARTE PARA OS DADOS SEM  FASE /////////////////////////////////
#caminho para leitura (alterar aqui quando alterar o ruido e vice versa)
# caminhoArquivoSemFase = 'C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimo/Dados Estimação/DadosOcupacao_50.txt'

caminhoArquivoSemFase = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimo/Dados Estimação/DadosOcupacao_50.txt"

# Ler as 7 primeiras colunas (sinais)
dataOcupacaoFaseAmostras = np.loadtxt(caminhoArquivoSemFase, usecols=range(7))
# Ler apenas a amplitude do arquivo 
dataOcupacaoFaseAmplitude = np.loadtxt(caminhoArquivoSemFase, usecols=7)

# print("Data\n", dataOcupacaoFaseAmostras)
# print("DataAmplitude\n", dataOcupacaoFaseAmplitude)


primeiracoluna = dataOcupacaoFaseAmostras[:,0]
segundacoluna = dataOcupacaoFaseAmostras[:,1]
terceiracoluna = dataOcupacaoFaseAmostras[:,2]
quartacoluna = dataOcupacaoFaseAmostras[:,3]
quintacoluna = dataOcupacaoFaseAmostras[:,4]
sextacoluna = dataOcupacaoFaseAmostras[:,5]
setimacoluna = dataOcupacaoFaseAmostras[:,6]
plt.hist(primeiracoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(primeiracoluna)))
plt.legend(loc=0)
plt.title("Primeira Coluna")
plt.show()

plt.hist(segundacoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(segundacoluna)))
plt.legend(loc=0)

plt.title("Segunda Coluna")
plt.show()

plt.hist(terceiracoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(terceiracoluna)))
plt.title("Terceira Coluna")
plt.legend(loc=0)
plt.show()

plt.hist(quartacoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(quartacoluna)))
plt.legend(loc=0)
plt.title("Quarta Coluna")
plt.show()

plt.hist(quintacoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(quintacoluna)))
plt.legend(loc=0)
plt.title("Quinta Coluna")
plt.show()


plt.hist(sextacoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(sextacoluna)))
plt.legend(loc=0)
plt.title("Sexta Coluna")
plt.show()


plt.hist(setimacoluna, bins=50, alpha=0.7, histtype='step', label="janelamento="+str(n_janelamento)+" media="+str(np.mean(setimacoluna)))
plt.legend(loc=0)
plt.title("Setima Coluna")
plt.show()


def montarMatrizCovarianciaRuido(dataOcupacaoFaseAmostras):
    # Calcular a matriz de covariância do ruído
    cov_matrix_ruido = np.cov(dataOcupacaoFaseAmostras, rowvar=False)
    return cov_matrix_ruido


#Montar matriz de coeficientes
def montarMatrizCoeficientes(g, derivada_g, cov_matrix_ruido):
    n_janelamento = len(g)
    A = np.zeros((n_janelamento+3, n_janelamento+3)) # matriz A do sistema para encontrar os pesos

    # Preencher as primeiras n_janelamento casas (tanto linhas como colunas) da matriz A com a matriz de covariância do ruído
    for i in range(n_janelamento):
        for j in range(n_janelamento):
            A[i][j] = cov_matrix_ruido[i][j]

    for i in range(n_janelamento):
        A[i][n_janelamento] =-g[i] 
        A[i][n_janelamento+1] = -derivada_g[i]
        A[i][n_janelamento+2] = -1
        A[n_janelamento][i] = g[i]
        A[n_janelamento+1][i] = derivada_g[i]
        A[n_janelamento+2][i] = 1

    return A

def verificar_solucao(matrizA, matrizB):
    matrizAAmpliada = np.hstack((matrizA, matrizB))
    postoA = np.linalg.matrix_rank(matrizA)
    postoAAmpliada = np.linalg.matrix_rank(matrizAAmpliada)
    n = matrizA.shape[1]  # pega a quantidade de variáveis da matriz A
    if postoA != postoAAmpliada:
        return "Sistema não possui solução!"
    else:
        if postoA == n:
            solucao = np.linalg.solve(matrizA, matrizB)  
            return solucao
        if postoA < n:
            return "Sistema com múltiplas soluções"


cov_matrix_ruido = montarMatrizCovarianciaRuido(dataOcupacaoFaseAmostras)
# Montar a matriz A
A = montarMatrizCoeficientes(g, derivada_g, cov_matrix_ruido)


# Definir o vetor B
B = np.zeros((len(g) + 3, 1))
B[len(g)] = 1

# Verificar a solução do sistema linear
solucao_sistema = verificar_solucao(A, B)

# Extrair o vetor de pesos (w)
w = []
if type(solucao_sistema) != str:  
    for i in range(len(solucao_sistema)):
        #adiciona dinamicamente ao vetor de pesos
        if(i <= len(g) - 1):
            w.append(solucao_sistema[i][0])
else:
    print(solucao_sistema)

# print("Vetor de pesos:\n", w)
# print("Soma vetor pesos: ", sum(w))

def estimarAmplitude(dataOcupacaoFaseAmostras, pedestal, pesos):
    amplitude_estimada = []
    for i in range(len(dataOcupacaoFaseAmostras)):  # para cada linha
        soma = 0
        for j in range(len(pesos)):  # para cada coluna
            multiplicacao = (dataOcupacaoFaseAmostras[i][j] - pedestal) * pesos[j]
            soma += multiplicacao
        amplitude_estimada.append(soma)
    return amplitude_estimada

# Exemplo de uso da função
# Supondo que você já tenha matriz_Ocupacao_Amostras, pedestal e w

# Calcular amplitude estimada sem o k fold (tamanho total do conjunto de amostras da amplitude)
amplitude_estimada = estimarAmplitude(dataOcupacaoFaseAmostras, pedestal, w)

# print("Amplitude estimada: \n", amplitude_estimada)
# print("Tamanho amplitude estimada:", len(amplitude_estimada))
# print("Tamanho amplitude real:", len(dataOcupacaoFaseAmplitude))


####################################################### PROCESSAMENTO DOS DADOS PELO KFOLD ##########################################
k = 100
kf = KFold(n_splits=k)
mediaKfold = []
desvioPadraoKfold =[]
for fold, (train_index, test_index) in enumerate(kf.split(dataOcupacaoFaseAmostras)):
    matrizAmostrasTreino, matrizAmostrasTeste = dataOcupacaoFaseAmostras[train_index,:], dataOcupacaoFaseAmostras[test_index,:]
    amplitudeAmostrasTreino, amplitudeAmostrasTestes = dataOcupacaoFaseAmplitude[train_index], dataOcupacaoFaseAmplitude[test_index]
    covMatrizRuido = montarMatrizCovarianciaRuido(matrizAmostrasTreino)
    A_coeficientes = montarMatrizCoeficientes(g, derivada_g, covMatrizRuido)
    # Verificar a solução do sistema linear
    solucao_sistemaKFold = verificar_solucao(A_coeficientes, B)
    # Extrair o vetor de pesos (w)
    w_kfold = []
    if type(solucao_sistemaKFold) != str:  
        for i in range(len(solucao_sistemaKFold)):
            #adiciona dinamicamente ao vetor de pesos
            if(i <= len(g) - 1):
                w_kfold.append(solucao_sistemaKFold[i][0])
    else:
        print(solucao_sistemaKFold)
    erroEstimacaoKFold = []

    amplitude_estimadaTeste = estimarAmplitude(matrizAmostrasTeste, pedestal, w_kfold)
    for k in range(len(amplitude_estimadaTeste)):
        erroEstimacaoKFold.append(amplitude_estimadaTeste[k]- amplitudeAmostrasTestes[k])

    print("Fold:", fold)
    print("Pesos:", w_kfold)
    # Plotagem do histograma

    mediaKfold.append(np.mean(erroEstimacaoKFold))
    desvioPadraoKfold.append(np.std(erroEstimacaoKFold))
    plt.hist(erroEstimacaoKFold, bins=50, alpha=0.7, histtype='step', label="Fold = "+str(fold)+"média="+str(mediaKfold[-1]))

plt.xlabel('Erro de Estimação')
plt.ylabel('Frequência')
plt.title('Histograma do Erro de Estimação para Todos os Folds')
plt.legend(loc=0)
plt.grid(True)
plt.show()
# print("Media k fold", mediaKfold)
# print("Desvio padrao k fold", desvioPadraoKfold)

mediaDaMediaKFold = np.mean(mediaKfold)
desvioPadraoDoDesvioPadrao = np.std(desvioPadraoKfold)
mediaDesvioPadraoKFold = np.mean(desvioPadraoKfold)

# print("Media da Media: ", mediaDaMediaKFold)
# print("Desvio padrao do desvio padrao: ", desvioPadraoDoDesvioPadrao)

