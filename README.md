module load nvhpc

nvcc -O3 first.cu -o first -arch=sm_80 

nvcc -O3 prac2.cu -o prac -arch=sm_80 -lcurand -I.

srun -p gpu --mem-per-cpu=2000 --time=00:01:00 --ntasks=1 --gres=gpu:1 --reservation=<redacted> ./first
