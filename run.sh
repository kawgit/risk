#!/bin/bash -l
#$ -N risk_seqtrain
#$ -l h_rt=10:00:00
#$ -pe omp 1
#$ -l mem_per_core=4G
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
java -cp "./lib/*:src" edu.bu.pas.risk.SequentialTrain pas.risk.agent.RiskQAgent random -x 100 -g .9 -t 10 -v 1 2>&1