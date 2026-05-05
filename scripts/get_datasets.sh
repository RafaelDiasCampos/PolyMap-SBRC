#!/usr/bin/env bash

# Bot-IoT
clear
mkdir -p dataset/bot_iot
echo "Bot-IoT"
echo "Acesse a URL https://unsw-my.sharepoint.com/:f:/g/personal/z5131399_ad_unsw_edu_au/EjlBDf2KODxPgXmqbO3MxxsBBVARCKZxGUG47OiFHb_AnQ , entre na pasta 'Dataset/5%/All features', faça download dos arquivos 'UNSW_2018_IoT_Botnet_Full5pc_1.csv', 'UNSW_2018_IoT_Botnet_Full5pc_2.csv', 'UNSW_2018_IoT_Botnet_Full5pc_3.csv' e 'UNSW_2018_IoT_Botnet_Full5pc_4.csv' e os mova para a pasta dataset/bot_iot"
read -p "Aperte ENTER após copiar os arquivos para prosseguir para o próximo dataset"

# Dataset já incluso no repositório
# # TON_IoT
# clear
# mkdir -p dataset/ton_iot
# echo "TON_IoT"
# echo "Acesse a URL https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW , entre na pasta 'Train_Test_datasets/Train_Test_Network_datasets', faça download do arquivo 'train_test_network.csv' e o mova para a pasta dataset/ton_iot"
# read -p "Aperte ENTER após copiar os arquivos para prosseguir para o próximo dataset"

# NSL-KDD
clear
mkdir -p dataset/nsl-kdd
echo "NSL-KDD"
echo "Acesse a URL https://www.kaggle.com/datasets/hassan06/nslkdd , faça download dos arquivos 'KDDTrain+.arff' e 'KDDTest+.arff' e os mova para a pasta dataset/nsl-kdd"
read -p "Aperte ENTER após copiar os arquivos para prosseguir para o próximo dataset"

# CTU-13
clear
mkdir -p dataset/ctu-13
echo "CTU-13"
echo "Realizando download do dataset CTU-13, aguarde..."
wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow -O dataset/ctu-13/capture20110818.binetflow.csv
wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow -O dataset/ctu-13/capture20110818-2.binetflow.csv

# Verify checksums
sha256sum -c scripts/datasets.sha256

if [ $? -ne 0 ]; then
    echo "Erro: Verificação de integridade falhou. Por favor, verifique os arquivos baixados."
    exit 1
else
    echo "Todos os arquivos foram verificados com sucesso."
fi
