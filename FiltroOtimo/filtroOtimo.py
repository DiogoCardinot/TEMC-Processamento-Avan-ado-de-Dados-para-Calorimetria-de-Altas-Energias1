import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
  
# pulso de referencia
g= [0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424]

# # Valores para o eixo x (tempo(ns))
# tempo = [i * 25 for i in range(len(g))]

# # Plotar
# plt.plot(tempo, g, color="C0", marker='o', linestyle='-')
# plt.xlabel('Tempo(ns)')
# plt.ylabel(r'$g$')
# plt.title(r'Plotagem de g em relação ao tempo')

# plt.show()

# derivada do pulso de referencia
dg = [0.00004019, 0.00333578, 0.03108120, 0.00000000, -0.02434490, -0.00800683, -0.00243344]

# numero de janelamento do pulso de referencia
n_janelamento = len(g)
# print("Numero de n_janelamento", len(g))


#////////////////////////////////////////// LEITURA DOS ARQUIVOS ///////////////////////////////////////////////
# Ler os dados do arquivo de ruido
data = np.loadtxt('C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/RuidoOcupacao_50.txt')

#ler os dados de ruido notebook
# data = np.loadtxt('C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/RuidoOcupacao_0.txt')


#///////////////////////////////////// PARTE PARA OS DADOS SEM  FASE /////////////////////////////////
#caminho para leitura (alterar aqui quando alterar o ruido e vice versa)
caminhoArquivoSemFase = 'C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/DadosOcupacao_50.txt'


# Ler as 7 primeiras colunas (sinais)
dataOcupacaoFaseAmostras = np.loadtxt(caminhoArquivoSemFase, usecols=range(7))
# Ler apenas a amplitude do arquivo 
dataOcupacaoFaseAmplitude = np.loadtxt(caminhoArquivoSemFase, usecols=7)

# print("Data\n", dataOcupacaoFaseAmostras)
# print("DataAmplitude\n", dataOcupacaoFaseAmplitude)


#///////////////////////// LER DADOS DE OCUPAÇÃO DO CSV ///////////////////////////////
# Definir listas para armazenar os dados
sinal = []
amplitudeCSV = []

# Caminho para o arquivo CSV
caminho_arquivo_csv = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/ocupacao_50.csv"

# Ler o arquivo CSV
with open(caminho_arquivo_csv, newline='') as csvfile:
    leitor = csv.reader(csvfile, delimiter=',')
    for linha in leitor:
        # Adicionar as 7 primeiras colunas à lista primeiras_colunas
        sinal.append([float(valor) for valor in linha[:7]])
        # Adicionar a oitava coluna à lista oitava_coluna
        amplitudeCSV.append(float(linha[7]))

# for i in range(10):
#      print(amplitudeCSV[i])

# for i in range(10):
#      print("sinal", sinal[i])

# transformar os dados em 7 colunas (ruido)
data = data.reshape(-1, 7)

# Calcular a matriz de covariância do ruido
cov_matrix = np.cov(data, rowvar=False)
#Matriz de covariancia com ruido branco
cov_MatrizID = np.eye(n_janelamento)
# print("Matriz de covariancia com ruido branco", cov_MatrizID)
# print("Matriz de Covariância\n:")
# print(cov_matrix)

# matriz A do sistema vazia
A = np.zeros((n_janelamento+3, n_janelamento+3))
A_ID = np.zeros((n_janelamento+3, n_janelamento+3))

# preenche as primeiras 7 casas(tanto linhas como colunas) da matriz A com a matriz de covariancia do ruido
for i in range(n_janelamento):
    for j in range(n_janelamento):
        A[i][j] = cov_matrix[i][j]
        A_ID[i][j]= cov_MatrizID[i][j]

# preenche as outras linhas e colunas da matriz A do sistema (para matriz de covariancia com ruido de fundo)
for j in range(n_janelamento):
        A[j][n_janelamento] = -g[j] # preenche a oitava coluna da matriz com o pulso de referência negativo (até a setima linha)
        A[n_janelamento][j] = g[j] # preenche a oitava linha da matriz com o pulso de referência positivo (até a setima coluna)
        A[j][n_janelamento+1] = -dg[j] # preenche a nona coluna com - derivada do pulso de referencia (até a setima linha)
        A[n_janelamento+1][j] = dg[j] # prenche a nona linha com a derivada do pulso de referencia(até a setima coluna)
        A[j][n_janelamento+2] = -1 # preenche a décima coluna, até a setima linha, com -1
        A[n_janelamento+2][j] = 1 # preenche a décima linha, até a sétima coluna, com 1

# preenche as outras linhas e colunas da matriz A do sistema (para matriz de covariancia com ruido de fundo)
for j in range(n_janelamento):
        A_ID[j][n_janelamento] = -g[j] # preenche a oitava coluna da matriz com o pulso de referência negativo (até a setima linha)
        A_ID[n_janelamento][j] = g[j] # preenche a oitava linha da matriz com o pulso de referência positivo (até a setima coluna)
        A_ID[j][n_janelamento+1] = -dg[j] # preenche a nona coluna com - derivada do pulso de referencia (até a setima linha)
        A_ID[n_janelamento+1][j] = dg[j] # prenche a nona linha com a derivada do pulso de referencia(até a setima coluna)
        A_ID[j][n_janelamento+2] = -1 # preenche a décima coluna, até a setima linha, com -1
        A_ID[n_janelamento+2][j] = 1 # preenche a décima linha, até a sétima coluna, com 1


#matriz dos resultados
B = np.zeros((n_janelamento+3,1))
#o único termo que deve ser 1 (nesse caso, o oitavo)
B[n_janelamento] = 1


#funcao para verificar se o problema tem solucao ou nao
def VerificaSolucao(matrizA, matrizB):
        matrizAAmpliada = np.hstack((matrizA, matrizB))
        # print(matrizAAmpliada)
        postoA = np.linalg.matrix_rank(matrizA)
        postoAAmpliada = np.linalg.matrix_rank(matrizAAmpliada)
        n= matrizA.shape[1] #pega a quantidade de variáveis da matriz A
        if(postoA != postoAAmpliada):
                print("Sistema nao possui solucao!")
        else:
           if(postoA == n):
                solucao = np.linalg.solve(matrizA, matrizB)  
                return solucao
           if(postoA < n):
                 return "Sistema com multiplas solucoes"

#vetor contendo as soluções do sistema linear para matriz de covariancia com ruido de fundo
solucaoSistema = VerificaSolucao(A,B)
#vetor contendo as soluções do sistema linear para matriz de covariancia com ruido branco
solucaoSistemaRuidoBranco = VerificaSolucao(A_ID,B)

#vetor contendo os pesos da solucao do sistema
w = [] #para ruido não branco
w_ruidoBranco = [] #para ruido branco
# imprimir as solucoes 
for i in range(len(solucaoSistema)):
        if(i<=6):
            print(r'$w_['+str(i)+']$= ', solucaoSistema[i][0])
            w.append(solucaoSistema[i][0])
        if(i==7):
            print(r'$lambda$ = ', solucaoSistema[i][0])
        if(i==8):
            print(r'$epsilon$ = ', solucaoSistema[i][0])
        if(i==9):
            print(r'$kappa$ = ', solucaoSistema[i][0])
 
# print("Vetor de pesos", w)
# print("Soma do vetor de pesos com ruido de fundo: ",sum(w))

# imprimir as solucoes 
for i in range(len(solucaoSistemaRuidoBranco)):
        if(i<=6):
            w_ruidoBranco.append(solucaoSistemaRuidoBranco[i][0])

# print("Pesos ruido branco: ", w_ruidoBranco)
# print("Soma do vetor de pesos com ruido brancos:", sum(w_ruidoBranco))


multiplicacao=0 #armazenar a multiplicacao da linha para dados de ocupacao.txt e ruido nao branco
multiplicacaoRuidoBranco=0 #armazenar a multiplicacao da linha para dados de ocupacao.txt e ruido branco
multiplicacaoCSV=0 #armazenar a multiplicacao da linha para dados de ocupacao.csv e ruido nao branco

amplitude_estimada = [] #armazenar a amplitude dados de ocupacao.txt e ruido nao branco
amplitude_estimada_ruido_branco= [] #armazenar a amplitude dados de ocupacao.txt e ruido branco
amplitude_estimada_csv = [] #armazenar a amplitude dados de ocupacao.csv e ruido nao branco

#percorrer todas as amostras (100.000)
for i in range(len(dataOcupacaoFaseAmostras)):
    soma=0  #armazenar a soma das linhas para dados de ocupacao.txt e ruido nao branco
    somaRuidoBranco = 0  #armazenar a soma das linhas para dados de ocupacao.txt e ruido branco
    #percorrer as colunas de cada amostra (7 colunas)
    for k in range(n_janelamento):
        #para cada linha, multiplica a amostra de k com o peso de k
        multiplicacao = dataOcupacaoFaseAmostras[i][k] * w[k] #para a matriz de covariância dos dados(.txt) com ruido nao branco
        multiplicacaoRuidoBranco = dataOcupacaoFaseAmostras[i][k]*w_ruidoBranco[k] #para matriz de covariancia com ruido branco
        #adiciona essa multiplicacao para cada
        soma += multiplicacao
        somaRuidoBranco += multiplicacaoRuidoBranco
    #adiciona a soma de cada linha como amplitude estimada    
    amplitude_estimada.append(soma) #dados(.txt) ruido nao branco
    amplitude_estimada_ruido_branco.append(somaRuidoBranco) #dados(.txt) ruido branco

#percorrer todas as amostras para o csv
for i in range(len(amplitudeCSV)):
    somaCSV = 0  #armazenar a soma das linhas para dados de ocupacao.csv e ruido nao branco
    #percorrer as colunas de cada amostra (7 colunas)
    for k in range(n_janelamento):
        multiplicacaoCSV = sinal[i][k]*w[k] #para matriz de covariancia dos dados(.csv) com ruido nao branco
        #adiciona essa multiplicacao para cada
        somaCSV += multiplicacaoCSV
    amplitude_estimada_csv.append(somaCSV) #dados(.csv) ruido nao branco


erroEstimacaoAmplitude = [] #vetor para armazenar os erros na estimação da amplitude
erroEstimacaoAmplitudeRuidoBranco=[]
erroEstimacaoCSV =[]

#calcular o erro entre a amplitude estimada e a amplitude verdadeira para o txt
for i in range(len(dataOcupacaoFaseAmplitude)):
      erroEstimacaoAmplitude.append(dataOcupacaoFaseAmplitude[i]-amplitude_estimada[i])
      erroEstimacaoAmplitudeRuidoBranco.append(dataOcupacaoFaseAmplitude[i]-amplitude_estimada_ruido_branco[i])
      
#calcular o erro entre a amplitude estimada e a amplitude verdadeira para o csv
for j in range(len(amplitude_estimada_csv)):
     erroEstimacaoCSV.append(amplitudeCSV[i]- amplitude_estimada_csv[i])

mediaErroEstimacao = np.mean(erroEstimacaoAmplitude)#média do erro de estimacao da amplitude
desvioPadraoErroEstimacao = np.std(erroEstimacaoAmplitude)#desvio padrao do erro de estimacao da amplitude
# print("Media Erro de Estimacao Amplitude Ruido Nao Branco .txt: ", mediaErroEstimacao)
# print("Desvio Padrao do Erro de Estimacao Amplitude Ruido Nao Branco .txt: ", desvioPadraoErroEstimacao)

mediaErroEstimacaoRuidoBranco = np.mean(erroEstimacaoAmplitudeRuidoBranco)
desvioPadraoErroEstimacaoRuidoBranco = np.std(erroEstimacaoAmplitudeRuidoBranco)
# print("Media Erro de Estimacao Amplitude Ruido Branco .txt: ", mediaErroEstimacaoRuidoBranco)
# print("Desvio Padrao do Erro de Estimacao Amplitude Ruido Branco .txt: ", desvioPadraoErroEstimacaoRuidoBranco)

mediaErroEstimacaoCSV = np.mean(erroEstimacaoCSV)
desvioPadraoErroEstimacaoCSV = np.std(erroEstimacaoCSV)
# print("Media Erro de Estimacao Amplitude CSV: ", mediaErroEstimacaoCSV)
# print("Desvio Padrão do Erro de Estimacao Amplitude CSV: ", desvioPadraoErroEstimacaoCSV)


#pega a porcentagem de ocupação para o sinal de ocupação .txt
numero = re.search(r'DadosOcupacao_(\d+)', caminhoArquivoSemFase).group(1)

#pega a porcentagem de ocupação para o sinal de ocupação .csv
numeroCSV = re.search(r'ocupacao_(\d+)', caminho_arquivo_csv).group(1)


mediaNB =[0.5173990704326287,1.4372111620720665,1.2815573899985069,1.143246588147817,1.3571329065674578, 1.3268466345663443,1.3266673894345413,1.2739387558707407,1.320658233108811,1.3272052749065841,1.3612691735205082] #eixo y
desvioPadraoNB =[3.3300792664087404,16.60005863160284,21.748460386617666,25.578718332596416,27.889982087497796,30.227737338892556,31.754501137693165,33.265568665071704,33.94813035679712,34.83263474121959,34.68191389350445] #eixo y
mediaB = [0.5162356080418578,0.6599428055167468,0.44762781655591927,0.3811614922032344,0.47858087014656087,0.4545305692510275, 0.6173242896120288,0.535819597907426,0.48197910790175746,0.4292431520668684,0.6184531200084465] #eixo y
desvioPadraoB = [3.329651873180055,19.619333699502263,26.861334243532802,31.962433686872274, 35.42182401499204,38.52799893789432,40.77301234067603,42.463728560356124,43.63750158605719,44.70187037381496,44.533153884651114] #eixo y
ocupacao = [0,10,20,30,40,50,60,70,80,90,100] #eixo x


# Plotar o histograma do erro de estimação com preenchimento

#dados do txt
plt.hist(erroEstimacaoAmplitude, bins=1000, label=f"Estimação com ruído de fundo ($\mu$={mediaErroEstimacao:.3f}, $\sigma$={desvioPadraoErroEstimacao:.3f}).") #histograma dados(.txt) com ruido não branco
plt.hist(erroEstimacaoAmplitudeRuidoBranco, bins=1000, histtype='step', label=f"Estimação com ruído branco ($\mu$={mediaErroEstimacaoRuidoBranco:.3f}, $\sigma$={desvioPadraoErroEstimacaoRuidoBranco:.3f}).") #histograma dados(.txt) com ruido branco

#dados do csv
# plt.hist(erroEstimacaoCSV, bins=1000, label="Estimação com ruído de fundo.") #histograma dados(.csv) com ruido não branco

# Plotar o histograma do erro associado ao ruído branco sem preenchimento
plt.xlabel('Erro de Estimação')
plt.ylabel('Frequência')
plt.title('Histograma de Erro de Estimação de Amplitude')
# Adicionar legenda com título
#legend para txt
plt.legend(title=rf"{numero}% Ocupação", loc=0) #legenda para dados .txt(erro e sinal)
#legend para csv
# plt.legend(title=rf"{numeroCSV}% Ocupação ($\mu$={mediaErroEstimacaoCSV:.3f}, $\sigma$={desvioPadraoErroEstimacaoCSV:.3f})", loc=0) #legenda para dados .txt(erro) e .csv(sinais)
plt.grid(True)
plt.show()




# Criar um mapa de calor da matriz de covariância
plt.figure(figsize=(10, 8))
# annot=True adiciona valores numéricos nas células do heatmap.
# cmap='coolwarm' define a paleta de cores utilizada.
# fmt=".5f" formata os valores numéricos com duas casas decimais.
# linewidths=.5 controla a espessura das linhas de separação entre as células.
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".5f")
# Personalizar os marcadores e rótulos dos eixos
plt.xticks(np.arange(0.5, cov_matrix.shape[1] + 0.5, 1), np.arange(1, cov_matrix.shape[1] + 1, 1))
plt.yticks(np.arange(0.5, cov_matrix.shape[0] + 0.5, 1), np.arange(1, cov_matrix.shape[0] + 1, 1))
# Mover os valores do eixo x para o topo da figura
plt.gca().xaxis.set_ticks_position('top')
# Adicionar uma legenda explicativa
plt.title(f"Matriz de Covariância do Ruído ({numero}% Ocupação)")
plt.show()


#////////////////// PLOTAR OS VETORES DE PESO /////////////////////
plt.figure(figsize=(10, 6)) 
plt.plot(w, label='Ruído Não Branco', color='black')
plt.plot(w_ruidoBranco, label='Ruído Branco', color='purple')  
plt.xlabel('Amostra') 
plt.ylabel('Valor do Peso')  
plt.title(f"Comparação entre os Pesos Ruído Não Branco e Ruído Branco ({numero}% Ocupação)")
plt.grid(True)
plt.legend()  # Adiciona a legenda
plt.show()  # Mostra o gráfico

#///////////// PLOTAR GRÁFICO DA COMPARAÇÃO ENTRE AS MÉDIAS RUIDO BRANCO E NÃO BRANCO ////////////////
plt.plot(ocupacao, mediaB, label="Média Ruído Branco", color='purple')
plt.plot(ocupacao, mediaNB, label="Média Ruído Não Branco", color='black')
plt.title(r'$Média \times Ocupacão$')
plt.xlabel(r'$Ocupação (\%)$')
plt.ylabel(r'$Média (\mu)$')
plt.xticks(range(0, 101, 10), range(0, 101, 10))
plt.legend(loc=0)
plt.show()

#///////////// PLOTAR GRÁFICO DA COMPARAÇÃO ENTRE AS DISPERSÕES RUIDO BRANCO E NÃO BRANCO ////////////////
plt.plot(ocupacao, desvioPadraoB, label="Dispersão Ruído Branco", color='purple')
plt.plot(ocupacao, desvioPadraoNB, label="Dispersão Ruído Não Branco", color='black')
plt.title(r'$Dispersão \times Ocupacão$')
plt.xlabel(r'$Ocupação (\%)$')
plt.ylabel(r'$Dispersão$')
plt.xticks(range(0, 101, 10), range(0, 101, 10))
plt.legend(loc=0)
plt.show()