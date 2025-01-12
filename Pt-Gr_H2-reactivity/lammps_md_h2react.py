import os
import sys
import glob
import numpy as np
from lammps import lammps
from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell
from mpi4py import MPI


#####################################
## Parameters
#####################################
N_H2 = 500               # Number of H2 molecules 
coverage = 0.25           # Pt coverage ratio as fraction of 1 ML
min_dist = 4.0           # Minimum initial distance in Å between new and base H2 molecules
initial_z_offset = 5.0   # Initial z-height for the first Pt atoms above graphene
max_z_offset = 20.0      # Maximum z-height offset from graphene

N_steps = int(1e6)       # Number of MD steps per batch
temp = 300.0             # Temperature in Kelvin
temp_init = 300.0        # Initial temperature
time_step = 0.001        # Time step in ps (1 fs)

chunk_size = 10000        # Chuck size to run before checking for the stop file to quit

#####################################
## MD
#####################################
# Initialize MPI and get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Read the initial_structure
initial_structure = read('initial_structure.extxyz')

# Write initial structure
write("initial.dat", initial_structure, format="lammps-data", atom_style='atomic')

# Initialize LAMMPS and set up the simulation
lmp = lammps(comm=comm)
lmp.command("units metal")
lmp.command("dimension 3")
lmp.command("boundary p p f")
lmp.command("newton on")
lmp.command("atom_style atomic")
lmp.command("read_data 'initial.dat'")

# Define Masses
lmp.command("mass 1 12.011")    # Carbon
lmp.command("mass 2 1.008")     # Hydrogen
lmp.command("mass 3 195.084")   # Platinum

# **Define Groups**
lmp.command("group graphene type 1")
lmp.command("group hydrogen type 2")
lmp.command("group platinum type 3")
# Create a combined group for hydrogen and platinum
lmp.command("group hp_union union hydrogen platinum")

# Define groups based on external files
cluster_files = glob.glob('../cluster_*.txt')
for filename in cluster_files:
    cluster_id = filename.split('_')[1].split('.')[0]  # Extract cluster ID from filename
    try:
        with open(filename, 'r') as file:
            atom_ids = file.read().strip().split()
            atom_ids_str = ' '.join(atom_ids)
        lmp.command(f"group cluster_{cluster_id} id {atom_ids_str}")
        lmp.command(f"fix fix_cluster_momentum_{cluster_id} cluster_{cluster_id} momentum 1 linear 1 1 1")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Fix the Momentum of the Graphene Sheet
lmp.command("fix fix_graphene_momentum graphene momentum 1 linear 1 1 1")

# Apply a reflecting wall at some z value for both Pt and H atoms
lmp.command("fix upper_wall hp_union wall/reflect zhi 70.0")

# NNP potential
lmp.command("pair_style allegro")
lmp.command("pair_coeff * * /home/common/akram/graphene_pt/allegro/20d/results/run/deployed_model.pth C H Pt")

# Define computes
lmp.command("compute atomicenergies all pe/atom")
lmp.command("compute totalatomicenergy all reduce sum c_atomicenergies")

# Thermodynamics
lmp.command("thermo 50")
lmp.command("thermo_style custom step temp vol press pe ke etotal c_totalatomicenergy")
lmp.command("thermo_modify flush yes")

# Define dump with per-atom energies
lmp.command("dump 1 all custom 50 traj.lmp id type x y z vx vy vz fx fy fz c_atomicenergies")

# Set up restart files to be written periodically 
restart_dir = 'checkpoint'
if rank == 0:
    if not os.path.exists(restart_dir):
        os.makedirs(restart_dir)
comm.Barrier()
lmp.command(f"restart 10000 {os.path.join(restart_dir, 'step.*.restart')}")

# Energy Minimization
lmp.command("min_style fire")
lmp.command("min_modify dmax 0.08")  # Modify the max step size for the minimization
lmp.command("minimize 3.0e-5 3.0e-3 1000 1000")

# MD -- NVT with Temperature Ramping
random_seed = 12345 + int(coverage*100)  # Unique seed per batch
lmp.command(f"velocity all create {temp_init} {random_seed} mom yes rot yes")
lmp.command(f"fix 1 all nvt temp {temp_init} {temp} {time_step*100}")


# Run the simulation
lmp.command(f"timestep {time_step}")

# Parameters for chunked run
total_steps = N_steps
steps_completed = 0

while steps_completed < total_steps:
    steps_remaining = total_steps - steps_completed
    current_chunk = min(chunk_size, steps_remaining)
    lmp.command(f"run {current_chunk}")
    steps_completed += current_chunk

    # Check for stop file
    if os.path.exists('stop.lammps'):
        if rank == 0:
            print("Stop file detected. Exiting simulation.")
        break

# Write final structure in batch directory
lmp.command("write_data 'final.dat'")

# Close LAMMPS
lmp.close()
