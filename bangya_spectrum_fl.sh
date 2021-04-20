
# trainning visulization at https://wandb.ai/neutrinoliu/fedml_spectrum
wandb login b8de1ba7e50c8756b94a6ca7497e8e50b6c25830

GPU=0           # whether enable gpu
CLIENT_NUM=20   # total num of client
WORKER_NUM=10   # receive how many local weights each round
BATCH_SIZE=10   # batch size
DATASET="spectrum_gps"
DATA_PATH="./../../../data/spectrum"
MODEL="fnn_spectrum"
ROUND=10        # how many times global update
EPOCH=10        # how many local epoch each round
LR=0.001        # learning rate
OPT="adam"      # optimizer

# sample test and training data from GPS-power.dat
cd ./singleMachine
python genTrainTest.py $CLIENT_NUM
cd ..

# run fl simualtion
cd ./fedml_experiments/standalone/spectrum_avg

python3 ./spectrum_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR 

echo "[SpectrumPrediction] loss should mulitiply with 55.27 to retrieve a dB^2 unit"
echo "[SpectrumPrediction] all training done! heatmap can be checked at fedml_experiments/standalone/spectrum_avg/"

cd ./../../../


