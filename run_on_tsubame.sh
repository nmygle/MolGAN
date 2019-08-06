
#$ -cwd
#$ -l f_node=4
#$ -l h_rt=0:40:00
#$ -N test
#$ -o logs/
#$ -e logs/
. /etc/profile.d/modules.sh

module load cuda
module load nccl
module load openmpi

export PATH=$HOME/miniconda3/bin:$PATH
source activate tf12
#python example.py
#horovodrun -np 4 -H localhost:4 python example.py
mpirun -npernode 4 -np 16 -x LD_LIBRARY_PATH python example.py
