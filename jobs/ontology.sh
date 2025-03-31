#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --mem-per-cpu=20000

#SBATCH --ntasks=1                      # Number of MPI ranks
#SBATCH --cpus-per-task=4               # Number of cores per MPI rank 

#SBATCH -p short
#SBATCH --output=logs/%j.log     # Standard output and error log
#SBATCH -e logs/%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,ALL                    # Type of email notification- BEGIN, END,FAIL,ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu

#python /Users/aashnashah/Desktop/ssh_mount/SETPOINT/notebooks/get_ontology.py
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2008 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2010 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2012 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2014 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2016 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2018 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2020 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2022 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 5 --year_cutoff 2024 --data_dir ../data --output_dir ../results/setpoint_calculations

python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2008 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2010 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2012 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2014 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2016 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2018 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2020 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2022 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 5 --year_cutoff 2024 --data_dir ../data --output_dir ../results/setpoint_calculations

python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2008 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2010 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2012 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2014 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2016 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2018 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2020 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2022 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 30 --min_tests 3 --year_cutoff 2024 --data_dir ../data --output_dir ../results/setpoint_calculations

python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2008 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2010 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2012 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2014 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2016 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2018 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2020 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2022 --data_dir ../data --output_dir ../results/setpoint_calculations
python ../setpoint.py --min_gap 15 --min_tests 3 --year_cutoff 2024 --data_dir ../data --output_dir ../results/setpoint_calculations
