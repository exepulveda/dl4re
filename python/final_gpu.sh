#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=08:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1

module load Theano
module load CUDA/7.5.18
module load HDF5/1.8.16-foss-2015b

echo "FINAL BINARY MODEL"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test_keras.py 200 100 50


