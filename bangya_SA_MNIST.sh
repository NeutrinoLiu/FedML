wandb login b8de1ba7e50c8756b94a6ca7497e8e50b6c25830

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    echo "$expected != $actual"
    return 1
  fi
}

round() {
  printf "%.${2}f" "${1}"
}

# cd ./fedml_experiments/standalone/fedavg
# # small test
# # sh run_fedavg_standalone_pytorch.sh 0 2 2 4 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 1
# # sh run_fedavg_standalone_pytorch.sh 0 2 2 4 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 1 1 0.03 sgd 1

# # assert that, for full batch and epochs=1, the accuracy of federated training(FedAvg) is equal to that of centralized training
# echo "--------- start centralized training ----------"
# sh run_fedavg_standalone_pytorch.sh 0 1 1 61664  mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 0
# centralized_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")

# echo "---------- start federated training -----------"
# sh run_fedavg_standalone_pytorch.sh 0 1000 1000 -1 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd 0
# federated_full_train_acc=$(cat wandb/latest-run/files/wandb-summary.json | python -c "import sys, json; print(json.load(sys.stdin)['Train/Acc'])")

# assert_eq $(round $centralized_full_train_acc 3) $(round $federated_full_train_acc 3)

cd ./fedml_experiments/standalone/spectrum_avg
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 spectrum_gps ./../../../data/spectrum fnn_spectrum hetero 3 3 0.001 adam 0

cd ./../../../


