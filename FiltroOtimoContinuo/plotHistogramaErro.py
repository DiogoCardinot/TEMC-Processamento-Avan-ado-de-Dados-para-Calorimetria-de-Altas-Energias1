import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

pedestal = 30
n_janelamento = [7, 9, 11, 13, 15, 17, 19]
ocupacao = [0,10,20, 30, 40, 50, 60, 70, 80, 90, 100]

def estimarAmplitude(matriz_amostras, pedestal, pesos):
    amplitude_estimada = []
    for i in range(len(matriz_amostras)):  # para cada linha
        soma = 0
        for j in range(len(pesos)):  # para cada coluna
            multiplicacao = (matriz_amostras[i][j] - pedestal) * pesos[j]
            soma += multiplicacao
        amplitude_estimada.append(soma)
    return amplitude_estimada

for f in range(len(ocupacao)):
    mediaKfold = []  # Lista para armazenar as médias dos erros estimados de todos os janelamentos
    desvioPadraoKfold = []  # Lista para armazenar os desvios padrão dos erros estimados de todos os janelamentos

    for d in range(len(n_janelamento)):
        ############################################### CARREGAR INFORMAÇÕES DO PULSO DE REFERÊNCIA E DERIVADA ##################################################
        # nome_arquivo_saida = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/valores_g_derivada.txt"

        # notebook
        nome_arquivo_saida = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/valores_g_derivada.txt"
        g = None
        derivada_g = None

        # Abrir o arquivo e procurar pelo janelamento desejado
        with open(nome_arquivo_saida, 'r') as arquivo_leitura:
            for linha in arquivo_leitura:
                # Dividir a linha em partes separadas por espaço
                partes = linha.split()
                # Verificar se há elementos suficientes na lista partes
                if len(partes) >= n_janelamento[d] + 1:
                    # Se a primeira parte for igual ao número do janelamento desejado
                    if int(partes[0]) == n_janelamento[d]:
                        # Converter as partes relevantes para os vetores desejados
                        g = np.array(eval(' '.join(partes[1:n_janelamento[d] + 1])))
                        derivada_g = np.array(eval(' '.join(partes[n_janelamento[d] + 1:])))
                        break  # Parar de percorrer o arquivo após encontrar os dados

        # Verificar se os dados foram encontrados e carregados corretamente
        if g is not None and derivada_g is not None:
            # print("Dados de g carregados:\n", g)
            # print("Dados de derivada de g carregados:\n", derivada_g)
            print("\n\n")
        else:
            print("Dados para o janelamento especificado não foram encontrados ou estão incompletos.")

        #################################### LER AS AMOSTRAS E AS AMPLITUDES ASSOCIADAS DE ACORDO COM O JANELAMENTO ###################################################################

        # nome_arquivo_amostras = "C:/Users/diogo/Desktop/Diogo(Estudos)/Mestrado/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias1/FiltroOtimoContinuo/Dados_Ocupacoes/OC_" + str(
            # ocupacao[f]) + ".txt"

        # notebook
        nome_arquivo_amostras = "C:/Users/diogo/OneDrive/Área de Trabalho/TEMC-Processamento-Avan-ado-de-Dados-para-Calorimetria-de-Altas-Energias11/FiltroOtimoContinuo/Dados_Ocupacoes/OC_"+str(ocupacao[f])+".txt"

        def montarMatrizSinaisEAmplitude(nome_arquivo_amostras, n_janelamento):
            dados_amostras = np.genfromtxt(nome_arquivo_amostras, delimiter=",", skip_header=1)

            # Calcular o número de linhas para a matriz
            num_linhas = len(dados_amostras) - (n_janelamento - 1)

            # Inicializar a matriz para armazenar os blocos de amostras, com num_linhas linhas e n_janelamento colunas
            matriz_amostras = np.zeros((num_linhas, n_janelamento))

            # Preencher a matriz com os blocos de amostras
            for i in range(num_linhas):
                inicio = i  # começa no ponto atual
                fim = i + n_janelamento  # vai até o ponto atual + n_janelamento
                matriz_amostras[i] = dados_amostras[inicio:fim, 1]  # salva a linha na matriz

            # Calcular o índice do valor central em cada linha
            indice_central = n_janelamento // 2

            # Inicializar um array para armazenar as amplitudes associadas
            amplitude_real = np.zeros(num_linhas)

            # Preencher o array de amplitudes associadas
            for i in range(num_linhas):
                amplitude_real[i] = dados_amostras[i + indice_central, 2]

            return matriz_amostras, amplitude_real

        matriz_amostras, amplitude_real = montarMatrizSinaisEAmplitude(nome_arquivo_amostras, n_janelamento[d])


        def montarMatrizCovarianciaRuido(matriz_amostras):
            # Calcular a matriz de covariância do ruído
            cov_matrix_ruido = np.cov(matriz_amostras, rowvar=False)
            return cov_matrix_ruido


        # Montar matriz de coeficientes
        def montarMatrizCoeficientes(g, derivada_g, cov_matrix_ruido):
            A = np.zeros((n_janelamento[d] + 3, n_janelamento[d] + 3))  # matriz A do sistema para encontrar os pesos

            # Preencher as primeiras n_janelamento casas (tanto linhas como colunas) da matriz A com a matriz de covariância do ruído
            for i in range(n_janelamento[d]):
                for j in range(n_janelamento[d]):
                    A[i][j] = cov_matrix_ruido[i][j]

            for i in range(n_janelamento[d]):
                A[i][n_janelamento[d]] = -g[i]
                A[i][n_janelamento[d] + 1] = -derivada_g[i]
                A[i][n_janelamento[d] + 2] = -1
                A[n_janelamento[d]][i] = g[i]
                A[n_janelamento[d] + 1][i] = derivada_g[i]
                A[n_janelamento[d] + 2][i] = 1

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

        cov_matrix_ruido = montarMatrizCovarianciaRuido(matriz_amostras)
        # Montar a matriz A
        A = montarMatrizCoeficientes(g, derivada_g, cov_matrix_ruido)

        # Definir o vetor B
        B = np.zeros((len(g) + 3, 1))
        B[len(g)] = 1

        # Processamento dos dados pelo KFold
        k = 100
        kf = KFold(n_splits=k)
        erroEstimacaoKFold = []  # Lista para armazenar os erros estimados do KFold

        for fold, (train_index, test_index) in enumerate(kf.split(matriz_amostras)):
            matrizAmostrasTreino, matrizAmostrasTeste = matriz_amostras[train_index, :], matriz_amostras[test_index, :]
            amplitudeAmostrasTreino, amplitudeAmostrasTestes = amplitude_real[train_index], amplitude_real[test_index]
            covMatrizRuido = montarMatrizCovarianciaRuido(matrizAmostrasTreino)
            A_coeficientes = montarMatrizCoeficientes(g, derivada_g, covMatrizRuido)
            # Verificar a solução do sistema linear
            solucao_sistemaKFold = verificar_solucao(A_coeficientes, B)
            # Extrair o vetor de pesos (w)
            w_kfold = []
            if type(solucao_sistemaKFold) != str:
                for i in range(len(solucao_sistemaKFold)):
                    # adiciona dinamicamente ao vetor de pesos
                    if (i <= len(g) - 1):
                        w_kfold.append(solucao_sistemaKFold[i][0])
            else:
                print(solucao_sistemaKFold)

            # Calcular amplitude estimada para o conjunto de teste
            amplitude_estimadaTeste = estimarAmplitude(matrizAmostrasTeste, pedestal, w_kfold)

            # Calcular erro de estimativa para o conjunto de teste e armazenar na lista
            for k in range(len(amplitude_estimadaTeste)):
                erroEstimacaoKFold.append(amplitude_estimadaTeste[k] - amplitudeAmostrasTestes[k])
            mediaKfold.append(np.mean(erroEstimacaoKFold))
            desvioPadraoKfold.append(np.std(erroEstimacaoKFold))

        # Plotar o histograma apenas para o valor atual de janelamento
        plt.hist(erroEstimacaoKFold, bins=50, alpha=0.7, histtype='step', label=f"Janelamento: {n_janelamento[d]} ($\mu$= {mediaKfold[-1]:.3f},$\sigma$={desvioPadraoKfold[-1]:.3f} )")
        # print("Janelamento:", n_janelamento[d])
        # print("Ocupacao:", ocupacao[f])
        # print("Erro estimacao: \n", erroEstimacaoKFold)

    plt.xlabel("Erro Estimação")
    plt.ylabel("Frequência")
    plt.title("Ocupação: " + str(ocupacao[f]))
    plt.legend(loc=0)
    plt.show()

    # Cálculo da média das médias e do desvio padrão dos desvios padrão para toda a ocupação
    mediaDaMediaKFold = np.mean(mediaKfold)
    desvioPadraoDoDesvioPadrao = np.std(desvioPadraoKfold)
    mediaDesvioPadraoKFold = np.mean(desvioPadraoKfold)

    # Salvar esses dados ou fazer qualquer operação necessária