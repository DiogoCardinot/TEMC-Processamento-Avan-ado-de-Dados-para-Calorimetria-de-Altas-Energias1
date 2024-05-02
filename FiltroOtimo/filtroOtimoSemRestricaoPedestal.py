import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
  
# pulso de referencia
g= [0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424]

# derivada do pulso de referencia
dg = [0.00004019, 0.00333578, 0.03108120, 0.00000000, -0.02434490, -0.00800683, -0.00243344]

# numero de janelamento do pulso de referencia
n_janelamento = len(g)
# print("Numero de n_janelamento", len(g))

#////////////////////////////////////////// LEITURA DOS ARQUIVOS ///////////////////////////////////////////////
# Ler os dados do arquivo de ruido
# data = np.loadtxt('C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/RuidoOcupacao_50.txt')

#ler os dados de ruido notebook
data = np.loadtxt('C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/RuidoOcupacao_50.txt')

#///////////////////////////////////// PARTE PARA OS DADOS SEM  FASE /////////////////////////////////
#caminho para leitura (alterar aqui quando alterar o ruido e vice versa)
# caminhoArquivoSemFase = 'C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/DadosOcupacao_50.txt'
caminhoArquivoSemFase = 'C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias/FiltroOtimo/Dados Estimação/DadosOcupacao_50.txt'



# Ler as 7 primeiras colunas (sinais)
dataOcupacaoFaseAmostras = np.loadtxt(caminhoArquivoSemFase, usecols=range(7))
# Ler apenas a amplitude do arquivo 
dataOcupacaoFaseAmplitude = np.loadtxt(caminhoArquivoSemFase, usecols=7)

# print("Data\n", dataOcupacaoFaseAmostras)
# print("DataAmplitude\n", dataOcupacaoFaseAmplitude)


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
A_SemRestricaoPedestal = np.zeros((n_janelamento+2, n_janelamento+2))

# preenche as primeiras 7 casas(tanto linhas como colunas) da matriz A com a matriz de covariancia do ruido
for i in range(n_janelamento):
    for j in range(n_janelamento):
        A[i][j] = cov_matrix[i][j]
        A_SemRestricaoPedestal[i][j]= cov_matrix[i][j]

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
        A_SemRestricaoPedestal[j][n_janelamento] = -g[j] # preenche a oitava coluna da matriz com o pulso de referência negativo (até a setima linha)
        A_SemRestricaoPedestal[n_janelamento][j] = g[j] # preenche a oitava linha da matriz com o pulso de referência positivo (até a setima coluna)
        A_SemRestricaoPedestal[j][n_janelamento+1] = -dg[j] # preenche a nona coluna com - derivada do pulso de referencia (até a setima linha)
        A_SemRestricaoPedestal[n_janelamento+1][j] = dg[j] # prenche a nona linha com a derivada do pulso de referencia(até a setima coluna)


#matriz dos resultados
B = np.zeros((n_janelamento+3,1))
#o único termo que deve ser 1 (nesse caso, o oitavo)
B[n_janelamento] = 1

#matriz de resultado para sistema sem restricao de pedestal
B_SemRestricaoPedestal = np.zeros((n_janelamento+2,1))
B_SemRestricaoPedestal[n_janelamento] = 1

# print("Matriz A\n", A_SemRestricaoPedestal)
# print("Matriz B\n", B_SemRestricaoPedestal)

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
solucaoSistemaSemRestricaoPedestal = VerificaSolucao(A_SemRestricaoPedestal,B_SemRestricaoPedestal)

#vetor contendo os pesos da solucao do sistema
w = [] #para ruido não branco
w_SemRestricaoPedestal = [] #para ruido branco
# imprimir as solucoes 
for i in range(len(solucaoSistema)):
        if(i<=6):
            # print(r'$w_['+str(i)+']$= ', solucaoSistema[i][0])
            w.append(solucaoSistema[i][0])
        # if(i==7):
        #     print(r'$lambda$ = ', solucaoSistema[i][0])
        # if(i==8):
        #     print(r'$epsilon$ = ', solucaoSistema[i][0])
        # if(i==9):
        #     print(r'$kappa$ = ', solucaoSistema[i][0])
 
# print("Vetor de pesos", w)
# print("Soma do vetor de pesos com ruido de fundo: ",sum(w))

# imprimir as solucoes 
for i in range(len(solucaoSistemaSemRestricaoPedestal)):
        if(i<=6):
            w_SemRestricaoPedestal.append(solucaoSistemaSemRestricaoPedestal[i][0])

# print("Pesos sem restricao pedestal: ", w_SemRestricaoPedestal)
# print("Soma do vetor de pesos sem restricao pedestal:", sum(w_SemRestricaoPedestal))


multiplicacao=0 #armazenar a multiplicacao da linha para dados de ocupacao.txt e ruido nao branco
multiplicacaoSemRestricaoPedestal=0 #armazenar a multiplicacao da linha para dados de ocupacao.txt e ruido branco

amplitude_estimada = [] #armazenar a amplitude dados de ocupacao.txt e ruido nao branco
amplitude_estimada_sem_restricao_pedestal= [] #armazenar a amplitude dados de ocupacao.txt e ruido branco

#percorrer todas as amostras (100.000)
for i in range(len(dataOcupacaoFaseAmostras)):
    soma=0  #armazenar a soma das linhas para dados de ocupacao.txt e ruido nao branco
    somaRuidoBranco = 0  #armazenar a soma das linhas para dados de ocupacao.txt e ruido branco
    #percorrer as colunas de cada amostra (7 colunas)
    for k in range(n_janelamento):
        #para cada linha, multiplica a amostra de k com o peso de k
        multiplicacao = (dataOcupacaoFaseAmostras[i][k]-30) * w[k] #para a matriz de covariância dos dados(.txt) com ruido nao branco
        multiplicacaoSemRestricaoPedestal = (dataOcupacaoFaseAmostras[i][k]-30)*w_SemRestricaoPedestal[k] #para matriz de covariancia com ruido branco
        #adiciona essa multiplicacao para cada
        soma += multiplicacao
        somaRuidoBranco += multiplicacaoSemRestricaoPedestal
    #adiciona a soma de cada linha como amplitude estimada    
    amplitude_estimada.append(soma) #dados(.txt) ruido nao branco
    amplitude_estimada_sem_restricao_pedestal.append(somaRuidoBranco) #dados(.txt) ruido branco




erroEstimacaoAmplitude = [] #vetor para armazenar os erros na estimação da amplitude
erroEstimacaoAmplitudeSemRestricaoPedestal=[]

#calcular o erro entre a amplitude estimada e a amplitude verdadeira para o txt
for i in range(len(dataOcupacaoFaseAmplitude)):
      erroEstimacaoAmplitude.append(dataOcupacaoFaseAmplitude[i]-amplitude_estimada[i])
      erroEstimacaoAmplitudeSemRestricaoPedestal.append(dataOcupacaoFaseAmplitude[i]-amplitude_estimada_sem_restricao_pedestal[i])
      
mediaErroEstimacao = np.mean(erroEstimacaoAmplitude)#média do erro de estimacao da amplitude
desvioPadraoErroEstimacao = np.std(erroEstimacaoAmplitude)#desvio padrao do erro de estimacao da amplitude
# print("Media Erro de Estimacao Amplitude Ruido Nao Branco .txt: ", mediaErroEstimacao)
print("Desvio Padrao do Erro de Estimacao Amplitude Ruido Nao Branco .txt: ", desvioPadraoErroEstimacao)

mediaErroEstimacaoSemRestricaoPedestal = np.mean(erroEstimacaoAmplitudeSemRestricaoPedestal)
desvioPadraoErroEstimacaoSemRestricaoPedestal = np.std(erroEstimacaoAmplitudeSemRestricaoPedestal)
# print("Media Erro de Estimacao Amplitude Ruido Branco .txt: ", mediaErroEstimacaoSemRestricaoPedestal)
print("Desvio Padrao do Erro de Estimacao Amplitude Sem restricao de Pedestal .txt: ", desvioPadraoErroEstimacaoSemRestricaoPedestal)



#pega a porcentagem de ocupação para o sinal de ocupação .txt
numero = re.search(r'DadosOcupacao_(\d+)', caminhoArquivoSemFase).group(1)


desvioPadraoNB =[3.3300792664087404,16.60005863160284,21.748460386617666,25.578718332596416,27.889982087497796,30.227737338892556,31.754501137693165,33.265568665071704,33.94813035679712,34.83263474121959,34.68191389350445] #eixo y
desvioPadraoSemRestricaoPedestal = [2.4222813707807727,15.1493115965774, 20.156309657506256, 23.851462242295398,25.983134640915743, 28.22736137392555,  29.69106023580492,31.081681086701987,  31.795053984808746, 32.54608891556221, 32.354093736479356] #eixo y
ocupacao = [0,10,20,30,40,50,60,70,80,90,100] #eixo x



plt.hist(erroEstimacaoAmplitude, bins=1000, label=f"Estimação com restrição de pedestal ($\mu$={mediaErroEstimacao:.3f}, $\sigma$={desvioPadraoErroEstimacao:.3f}).") #histograma dados(.txt) com ruido não branco
plt.hist(erroEstimacaoAmplitudeSemRestricaoPedestal, bins=1000, histtype='step', label=f"Estimação sem restrição de pedestal ($\mu$={mediaErroEstimacaoSemRestricaoPedestal:.3f}, $\sigma$={desvioPadraoErroEstimacaoSemRestricaoPedestal:.3f}).") #histograma dados(.txt) com ruido branco

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


#////////////////// PLOTAR OS VETORES DE PESO /////////////////////
plt.figure(figsize=(10, 6)) 
plt.plot(w, label='Com restrição de pedestal', color='black')
plt.plot(w_SemRestricaoPedestal, label='Sem restrição de pedestal', color='purple')  
plt.xlabel('Amostra') 
plt.ylabel('Valor do Peso')  
plt.title(f"Comparação entre os Pesos utilizando e não a restrição do pedestal ({numero}% Ocupação)")
plt.grid(True)
plt.legend()  # Adiciona a legenda
plt.show()  # Mostra o gráfico

#///////////// PLOTAR GRÁFICO DA COMPARAÇÃO ENTRE AS DISPERSÕES RUIDO BRANCO E NÃO BRANCO ////////////////
plt.plot(ocupacao, desvioPadraoSemRestricaoPedestal, label="Dispersão Sem Restrição do Pedestal", color='purple')
plt.plot(ocupacao, desvioPadraoNB, label="Dispersão Com Restrição do Pedestal", color='black')
plt.title(r'$Dispersão \times Ocupacão$')
plt.xlabel(r'$Ocupação (\%)$')
plt.ylabel(r'$Dispersão$')
plt.xticks(range(0, 101, 10), range(0, 101, 10))
plt.legend(loc=0)
plt.show()