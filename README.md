# Curriculum Learning com ConvLSTM

Este projeto utiliza redes neurais ConvLSTM para previsão de precipitação baseada em dados de radar.

## Dataset e Instalação

Devido ao tamanho dos arquivos (tensores .pt e CSVs), a pasta `data/` não está incluída neste repositório.

**1. Download dos dados:**
Os dados estão disponíveis no Google Drive neste link: **https://drive.google.com/drive/folders/1fwuvmTznguZNgxuiCtjfi2s4TZ2_sRk6?usp=sharing**

**2. Organização das pastas:**
Após baixar, extraia os arquivos na raiz do projeto para que a estrutura fique **exatamente** assim:

```text
IC_IFSP/
├── data/     <-- Você deve colar a pasta aqui
│   ├── csvs/
│   │   ├── ...
│   └── tensores/
│       ├──...
├── models/
├── src/
├── .gitignore
└── main.py
```
## Estrutura de Arquivos

Abaixo está a descrição dos principais módulos e scripts do projeto contidos na pasta `models/ConvLSTM/train_eval_model/`:

### Modelagem e Dados

* **`models.py`**
    * Define a arquitetura da rede neural.
    * Contém as classes `ConvLSTMCell` (célula básica com convoluções), `ConvLSTM_Layer` (camada recorrente) e `MultiLayerConvLSTM` (arquitetura empilhada).
    * Inclui funcionalidades como Dropout, Layer Normalization e inicialização de pesos (Kaiming).
    * Possui a função `create_conv_lstm_model` para instanciar modelos facilmente.

* **`dataset.py`**
    * Responsável pelo gerenciamento e carregamento dos dados.
    * **`MultiFrameNextTimestampDataset`**: Cria pares de input (sequência de frames) e target (frame futuro).
    * **`NormalizedMultiFrameDataset`**: Gerencia a normalização dos dados e organiza as sequências para o treinamento.

* **`norm_denorm.py`**
    * Utilitários para transformação dos dados.
    * Implementa funções de normalização (ex: `log_norm` usando logaritmo) e suas respectivas inversas (`log_denorm`) para garantir que o modelo treine com dados estáveis e as avaliações sejam feitas na escala real (mm/h).

### Treinamento e Avaliação

* **`train_ConvLSTM.py`**
    * Script para treinar **um único modelo**.
    * Configura hiperparâmetros, carrega os dados, executa o loop de treinamento e validação, aplica *early stopping* e salva o melhor modelo.

* **`trainMultipleModels.py`**
    * Script de automação para treinar **múltiplos modelos sequencialmente**.
    * Configurado para treinar modelos com diferentes horizontes de previsão (ex: 10, 20, 30, ..., 60 minutos à frente) em lote.
    * Gera gráficos comparativos de perdas (losses) e métricas entre todos os modelos treinados.

* **`test_models.py`**
    * Script dedicado à **inferência e teste**.
    * Carrega pesos de modelos previamente treinados (`.pth`), executa previsões em dados de teste e gera visualizações detalhadas e métricas de desempenho.

* **`performance_mesure.py`**
    * Módulo de métricas e visualização.
    * Calcula estatísticas como MSE, MAE, RMSE, R², SSIM (Structural Similarity) e NSE (Nash-Sutcliffe Efficiency).
    * Gera gráficos comparativos (scatter plots, boxplots) e visualizações visuais dos frames (Input vs. Predição vs. Real).
