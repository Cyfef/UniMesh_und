#!/bin/bash
#SBATCH --job-name=data      # 作业名称
#SBATCH --output=/public/home/swun-caiy2/Chen/data.txt     # 指定输出文件名模板，%j代表作业ID
#SBATCH --ntasks=1              # 请求任务数（即进程数）
#SBATCH --cpus-per-task=4        # 每个任务分配的CPU核心数
#SBATCH --time=02:00:00         # 作业最大运行时间（hh:mm:ss）
#SBATCH --partition=cn      # 指定分区（队列）

# 作业实际执行的命令
echo "Job started on:"
date

# 这里是你的程序代码或命令
#./my_program arg1 arg2
python /public/home/swun-caiy2/Chen/download.py

echo "Job finished on:"
date