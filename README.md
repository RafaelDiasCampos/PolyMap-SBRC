# PolyMap

Este repositório engloba o código e resultados experimentais utilizados para a escrita do artigo "Evasão em Modelos de Detecção de Ameaças de Rede Usando Propriedades do Espaço de Decisão", aceito para publicação na 44ª edição do Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC 2026).
O PolyMap consiste em um método para evasão em sistemas de detecção de ameaças de redes a partir do mapeamento do espaço de decisão do modelo de classificação como um conjunto de politopos convexos.
Inicialmente, é realizado um mapeamento, amostras de tráfego malicioso podem ser modificadas para 

## Formato do repositório

Neste repositório, os resultados obtidos para os modelos de detecção e métodos de ataque podem ser encontrados na pasta `results`.
Os arquivos `.ipynb` na raíz do repositório foram utilizados para execução e avaliação dos diferentes métodos de ataque e, posteriormente, para analisar os resultados e desenhar gráficos.

## Resultados obtidos

### Taxa de sucesso e distância média obtidas em cada dataset
![Resultados gerais do ataque](results/attack_results.png)

### Taxa de sucesso X distância média obtidas em cada execução do dataset TON_IoT em redes FNN (esquerda) e SNN (direita).
![Resultados TON_IoT](results/attack_results_ton_iot.png)

### Taxa de sucesso X distância média obtidas em cada execução do dataset Bot-IoT em redes FNN (esquerda) e SNN (direita).
![Resultados Bot-IoT](results/attack_results_bot_iot.png)

### Taxa de sucesso X distância média obtidas em cada execução do dataset NSL-KDD em redes FNN (esquerda) e SNN (direita).
![Resultados NSL-KDD](results/attack_results_nsl_kdd.png)

### Taxa de sucesso X distância média obtidas em cada execução do dataset CTU-13 em redes FNN (esquerda) e SNN (direita).
![Resultados CTU-13](results/attack_results_ctu_13.png)

## Execução do código

Para executar o código, é necessário primeiramente treinar os modelos de evasão, e depois conduzir os ataques em cada modelo treinado.
Por fim, podem ser analisados os resultados obtidos.

### Treinamento dos modelos de decisão

Antes de iniciar o treinamento dos modelos de decisão, os dados de cada dataset devem ser adicionados à pasta `dataset`, conforme a seguinte estrutura utilizada:

- Bot-IoT: Adicionar arquivos `UNSW_2018_IoT_Botnet_Full5pc_1.csv`, `UNSW_2018_IoT_Botnet_Full5pc_2.csv`, `UNSW_2018_IoT_Botnet_Full5pc_3.csv` e `UNSW_2018_IoT_Botnet_Full5pc_4.csv` na pasta `dataset/bot_iot`.
- TON_IoT: Adicionar arquivo `train_test_network.csv` na pasta `dataset/ton_iot`.
- NSL-KDD: Adicionar arquivos `KDDTrain+.arff` e `KDDTest+.arff` na pasta `dataset/nsl-kdd`.
- CTU-13: Adicionar arquivos `capture20110818.binetflow.csv` e `capture20110818-2.binetflow.csv` na pasta `dataset/ctu-13`.

Em seguida, o notebook python `1 - Training classifcation models.ipynb` pode ser executado para fazer o carregamento de cada dataset e o treinamento dos modelos de detecção para cada um.
Os resultados obtidos são exibidos no notebook e armazenados no arquivo `results\classification_results.json`.

### Execução dos ataques

Após treinar os modelos de detecção, o notebook python `2 - Executing attack strategies.ipynb` pode ser executado para executar os métodos de ataque em cada modelo de detecção treinado anteriormente.
Os resultados obtidos são exibidos no notebook e armazenados no arquivo `results\attack_results.json`.

### Análise dos resultados

Depois de executar os ataques, o notebook `3 - Plotting results.ipynb` pode ser utilizado para gerar gráficos indicando as métricas obtidas por cada ataque contra os modelos de detecção.