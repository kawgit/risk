#!/bin/bash -l
#$ -N train_v1
#$ -l h_rt=10:00:00
#$ -pe omp 2
#$ -l mem_per_core=8G
#$ -j y
#$ -o /usr4/cs440/kawgit56/risk/logs

# Load Java
module load java/21.0.4

# Compile
javac -cp "./lib/*:src" @risk.srcs || {
  echo "Compilation failed!" >&2
  exit 1
}

# Train
java -cp "./lib/*:src" edu.bu.pas.risk.SequentialTrain pas.risk.agent.RiskQAgent random -x 100 -g .9 -t 100 -v 10 -i /usr4/cs440/kawgit56/risk/params/qFunction10.model --outOffset 11 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }'