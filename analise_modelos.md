# Análise de Modelos: Detecção de Câncer de Pele

Este documento consolida a explicação e a análise comparativa entre os modelos **Random Forest** e **Deep Learning (EfficientNet)** para o diagnóstico de lesões de pele.

## 1. O Vencedor Claro: Deep Learning
O modelo de **Deep Learning superou o Random Forest em praticamente todos os cenários**.
* **MCC (Matthews Correlation Coefficient):** O MCC Macro saltou de **0.409** no Random Forest para **0.610** no Deep Learning. Isso significa que as previsões passaram a ter uma correlação forte e robusta com a realidade.
* **Acurácia Balanceada:** Subiu de **69.7%** (RF) para **81.4%** (DL), o que indica que a EfficientNet lida muito melhor com o desbalanceamento das classes.

## 2. A Classe mais Crítica: Melanoma
O melanoma é o câncer de pele mais agressivo e fatal. 
* **Random Forest:** A Sensibilidade (TPR) para melanoma foi de apenas **0.558 (55,8%)**. Isso gerou um **FNR (Falsos Negativos) de 44,2%**. O modelo mandaria 4 em cada 10 pacientes com melanoma para casa achando que estão saudáveis.
* **Deep Learning:** A Sensibilidade (TPR) para o melanoma subiu para **0.733 (73,3%)**. A taxa de falsos negativos caiu quase pela metade. Observando a matriz de confusão, apenas **1** caso de melanoma foi previsto erroneamente como Nevo Benigno.
* **Comportamento Conservador:** O DL errou prevendo 21 Nevos como sendo Melanoma (Falsos Positivos). Na medicina, é muito melhor o modelo "se assustar" com uma pinta benigna e pedir uma biópsia (FP) do que ignorar um câncer real (FN).

## 3. As Curvas ROC e AUC
* **Random Forest:** As curvas estão muito dispersas. A AUC para *Benign Keratosis* foi de apenas 0.75, e a do Melanoma foi 0.81.
* **Deep Learning:** As curvas estão visivelmente mais compactas e empurradas para o topo. A AUC do Melanoma subiu para **0.89**, e lesões vasculares chegaram a **0.99**. Uma AUC de 0.89 no Melanoma prova uma excelente capacidade de distinguir um melanoma de uma lesão saudável.

## Conclusão
O **Random Forest** não consegue capturar a complexidade visual das lesões de forma eficiente, errando perigosamente em diagnósticos críticos.
O **Deep Learning Híbrido** prova seu valor ao cruzar as features da imagem bruta (extraídas pela EfficientNet), reduzindo drasticamente os piores erros médicos. Para uso em produção, o modelo de Deep Learning é, sem dúvidas, a melhor escolha viável.
