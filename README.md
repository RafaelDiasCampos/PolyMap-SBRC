# PolyMap

Este repositório engloba o código e resultados experimentais utilizados para a escrita do artigo "Evasão em Modelos de Detecção de Ameaças de Rede Usando Propriedades do Espaço de Decisão", aceito para publicação na 44ª edição do Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC 2026).

O PolyMap consiste em um método para evasão em sistemas de detecção de ameaças de redes a partir do mapeamento do espaço de decisão do modelo de classificação como um conjunto de politopos convexos.
Inicialmente, é realizado um mapeamento do espaço de decisão do modelo alvo utilizando uma amostra de tráfego normal como base e a modificando até que ela seja classificada como tráfego malicioso.
Esse processo é repetido para diversas amostras de tráfego normal, e o espaço encontrado é representado como um conjunto de politopos.
Durante a realização de um ataque de rede, amostras de tráfego malicioso podem ser modificadas para que elas estejam dentro de um dos politopos encontrados, buscando minimizar a distância entre a amostra original e a adversarial.

# Estrutura do readme.md

Os arquivos `*.ipynb` na raíz do repositório foram utilizados para execução e avaliação dos diferentes métodos de ataque e, posteriormente, para analisar os resultados e desenhar gráficos.

A implementação dos modelos de classificação de tráfego de rede e dos métodos de ataque podem ser econtrados na pasta `utils`.

A pasta `dataset` deve ser preenchida com os datasets utilizados para treinamento dos modelos de classificação.

A pasta `snapshots` é preenchida automaticamente com os estados dos modelos de detecção treinados ao executar os códigos.

Os resultados obtidos para os modelos de detecção e métodos de ataque podem ser encontrados na pasta `results`.

# Selos Considerados

Para estes artefatos, são considerados os seguintes selos, com base nos códigos e resultados apresentados:

- Artefatos Disponíveis (SeloD)
- Artefatos Funcionais (SeloF)
- Artefatos Sustentáveis (SeloS)
- Experimentos Reprodutíveis (SeloR)

# Informações básicas

Os códigos contidos neste repositório foram desenvolvidos em Python, com o aux´ilio de bibliotecas de terceiros. A seção a seguir descreve as dependências necessárias para configurar e executar os experimentos.

# Dependências

A seguir, são listadas as dependências necessárias para a execução do PolyMap.

## Software

- Sistema operacional Linux (execução não foi verificada em sistemas Windows ou MacOS).
- Git
- Python 3 (utilizada versão 3.14.3).
- Python Pip
- Jupyter Notebook
- Instalação de bibliotecas disponíveis no arquivo `requirements.txt`

## Hardware

### Execução dos experimentos

- CPU: Mínimo 4 núcleos.
- RAM: 16GB (execução parcial) ou 40GB para execução de todos os experimentos em todos os datasets.
- Armazenamento: 50GB para execução de todos os experimentos em todos os datasets.
- Placa de Vídeo: Recomendado placa de vídeo com suporte a CUDA e pelo menos 6GB de VRAM para execução de todos os experimentos.

### Análise dos resultados

- Sistema operacional Linux (execução não foi verificada em sistemas Windows ou MacOS).
- CPU: Sem restrição.
- RAM: 4GB.
- Armazenamento: 5GB.

# Preocupações com segurança

- Uso de bibliotecas de terceiros: O código apresentado faz uso de bibliotecas terceiras obtidas pelo gerenciador de pacotes do python (pip) e pelo site externo do PyTorch. Apesar de terem sido escolhidas apenas bibliotecas de ampla utilização e boa reputação, existem riscos intrínsicos da utilização de códigos de terceiros.

# Instalação

Esta seção descreve o processo de obtenção do repositório, instalação de dependências e configuração.
Inicialmente, verifique se as dependências listadas na seção [Dependências](#dependências) estão corretamente instaladas.

Em seguida, clone o repostirório:

```
git clone https://github.com/RafaelDiasCampos/PolyMap-SBRC
cd PolyMap-SBRC
```

Em seguida, instale as dependências:

```
./scripts/install_dependencies.sh
```

# Teste mínimo

Após instalar e configurar as dependências, o notebook Python `3 - Plotting results.ipynb` pode ser executado para validar o funcionamento das bibliotecas instaladas.

# Experimentos

A realização dos experimentos completos exige altos recursos computacionais e um tempo elevado de execução de múltiplos dias.
Dessa forma, também oferecemos uma versão reduzida que permite executar experimentos em aproximadamente 5-6 horas no dataset TON_IoT, utilizando uma fração de 30% de seus dados e com apenas 3 repetições (ao invés de 7).
Nesta seção, descrevemos o processo de executar esses experimentos.

## Versão reduzida

Para executar a versão reduzida dos experimentos, execute o script automatizado:

```
./scripts/execute_experiments_reduced.sh
```

Os resultados obtidos serão salvos na pasta `results` e podem ser visualizados para análise.

### Resultados esperados - versão reduzida

Para a versão reduzida dos experimentos, é esperado obter resultados similares aos obtidos no dataset TON_IoT na versão original, conforme demonstrado nessa seção:

#### Taxa de sucesso e distância média obtidas
![Resultados gerais do ataque](results/reduced/attack_results.png)

#### Taxa de sucesso vs. distância média obtidas nas redes FNN (esquerda) e SNN (direita).
![Resultados TON_IoT](results/reduced/attack_results_ton_iot.png)

## Versão completa

A realização dos experimentos completos consiste em múltiplas etapas.
Primeiramente, deve ser feito o treinamento dos modelos de detecção de ameaças de rede.
Em seguida, devem ser realizados os ataques nos modelos treinados.
Por fim, os resultados obtidos podem ser analisados e os gráficos gerados.
As seções a seguir descrevem a execução de cada etapa dos experimentos.

### Obtenção dos datasets

Para obter os datasets, execute o script criado e siga as instruções para fazer download dos arquivos e copiá-los para as pastas. Esse processo é um pouco manual.

```
./scripts/install_dependencies.sh
```

### Configuração dos parâmetros

Este repositório está configurado com os parâmetros utilizados durante a execução dos experimentos para a escrita do artigo.
Na configuração padrão, são criados e treinados 8 modelos FNN e 8 modelos SNN para a classificação de tráfego de rede para cada dataset, representando um total de 64 modelos.
Em seguida, para cada modelo treinado são executados os ataques Gen-AAL e IDSGAN 5 vezes, e o ataque PolyMap 1 vez, para comparação de resultados.
Esse processo demora um tempo significativo de execução de múltiplos dias devido à grande quantidade de dados em cada dataset e de repetições de cada experimento.

Caso seja desejável reduzir a quantidade de repetições dos experimentos, podem ser alterados os parâmetros `n_copies` e `n_trials` no arquivo `utils/parameters.py`.

### Treinamento dos modelos de decisão

Para treinar os modelos de decisão, pode ser executado o notebook Python `1 - Training classification models.ipynb`.
Os resultados obtidos são exibidos no notebook e armazenados no arquivo `results\classification_results.json`.

### Execução dos ataques

Após treinar os modelos de detecção, o notebook python `2 - Executing attack strategies.ipynb` pode ser executado para executar os métodos de ataque em cada modelo de detecção treinado anteriormente.
Os resultados obtidos são exibidos no notebook e armazenados no arquivo `results\attack_results.json`.

### Análise dos resultados

Depois de executar os ataques, o notebook `3 - Plotting results.ipynb` pode ser utilizado para gerar gráficos indicando as métricas obtidas por cada ataque contra os modelos de detecção.
Os gráficos gerados são exibidos no notebook e salvos na pasta `results`.

### Resultados esperados - versão completa

Esta seção contém os resultados experimentais obtidos pela execução dos métodos de ataque.

#### Taxa de sucesso e distância média obtidas em cada dataset
![Resultados gerais do ataque](results/attack_results.png)

#### Taxa de sucesso vs. distância média obtidas em cada execução do dataset TON_IoT em redes FNN (esquerda) e SNN (direita).
![Resultados TON_IoT](results/attack_results_ton_iot.png)

#### Taxa de sucesso vs. distância média obtidas em cada execução do dataset Bot-IoT em redes FNN (esquerda) e SNN (direita).
![Resultados Bot-IoT](results/attack_results_bot_iot.png)

#### Taxa de sucesso vs. distância média obtidas em cada execução do dataset NSL-KDD em redes FNN (esquerda) e SNN (direita).
![Resultados NSL-KDD](results/attack_results_nsl_kdd.png)

#### Taxa de sucesso vs. distância média obtidas em cada execução do dataset CTU-13 em redes FNN (esquerda) e SNN (direita).
![Resultados CTU-13](results/attack_results_ctu_13.png)

# LICENSE

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.