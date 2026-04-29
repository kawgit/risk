#!/bin/bash -l
#$ -N train_v3_against_heuristic
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
java -ea -cp "./lib/*:src" edu.bu.pas.risk.SequentialTrain pas.risk.agent.RiskQAgent pas.risk.agent.HeuristicAgent -x 100 -g .99 -t 100 -v 10 -c 10000000000000000 2>&1