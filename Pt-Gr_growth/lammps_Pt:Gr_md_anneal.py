import os
import sys
import numpy as np
from lammps import lammps
from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell
from mpi4py import MPI


# Parameters
coverage = 0.25  # Total desired coverage ratio as fraction of 1 ML
supercell_size = 60      # Size factor for the supercell
min_dist = 2.8           # Minimum initial distance in Å between new and base Pt atoms
initial_z_offset = 3.2   # Initial z-height for the first Pt atoms above graphene
max_z_offset = 12.0      # Maximum z-height offset from graphene

N_steps = int(1e6)       # Number of MD steps per batch
temp = 300.0             # Temperature in Kelvin
temp_init = 300.0        # Initial temperature
time_step = 0.002        # Time step in ps (2 fs)

chunk_size = 10000        # Chuck size to run before checking for the stop file to quit

initial_structure_path = '*****' 
# Set to None for the first run, then update with the path to the final structure of the previous (lower coverage) run


# Initialize MPI and get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#####################################
## Generate or read initial structure
#####################################
if initial_structure_path:
    initial_structure = read(initial_structure_path)
else:
    # Read graphene conventional unit cell
    struc_gr_conv = read("POSCAR_gr_conv")
    # Make graphene supercell
    initial_structure = make_supercell(struc_gr_conv, [[supercell_size, 0, 0],
                                                       [0, int(round(supercell_size / np.sqrt(3))), 0],
                                                       [0, 0, 1]])


# Calculate monolayer factor for Pt atoms
monolayer_factor = supercell_size * int(round(supercell_size / np.sqrt(3))) * (2.468/2.805)**2 * 2
N_pt_total = int(coverage * monolayer_factor)

# radius for local search
local_radius = 5.0
def get_local_max_z_graphene(x, y, initial_structure, radius=5.0):
    # Extract positions of carbon atoms
    c_positions = initial_structure[initial_structure.symbols == 'C'].positions
    # Compute distance in x and y directions, considering periodic boundaries
    dx = c_positions[:, 0] - x
    dy = c_positions[:, 1] - y
    # Apply periodic boundary conditions
    cell_lengths = initial_structure.cell.lengths()[:2]
    dx -= cell_lengths[0] * np.rint(dx / cell_lengths[0])
    dy -= cell_lengths[1] * np.rint(dy / cell_lengths[1])
    distances_xy = np.sqrt(dx**2 + dy**2)
    # Find nearby graphene atoms within the specified radius
    nearby_indices = np.where(distances_xy < radius)[0]
    if len(nearby_indices) == 0:
        # If no nearby atoms, use the global max_z_graphene
        return max_z_graphene
    else:
        # Return the maximum z-coordinate among nearby graphene atoms
        return c_positions[nearby_indices, 2].max()


# Add Pt atoms
for _ in range(N_pt_total):
    placed = False
    attempt = 0
    while not placed:
        attempt += 1
        if attempt > 10000:
            if rank == 0:
                print(f"Unable to place Pt atom after 10000 attempts. Exiting.")
            sys.exit(1)
        # Generate random x and y coordinates within the cell
        x, y = np.random.rand(2) * initial_structure.cell.lengths()[:2]
        # Compute local max_z_graphene based on nearby graphene atoms
        local_max_z_graphene = get_local_max_z_graphene(x, y, initial_structure, radius=local_radius)
        z = local_max_z_graphene + initial_z_offset
        # Set a local max_z limit for z adjustments
        local_max_z = local_max_z_graphene + max_z_offset

        new_atom = Atoms('Pt', positions=[[x, y, z]])

        # Check distances to all existing atoms
        all_positions = initial_structure.positions
        # Compute distances considering periodic boundaries
        delta = all_positions - new_atom.positions[0]
        delta[:, 0] -= initial_structure.cell.lengths()[0] * np.rint(delta[:, 0] / initial_structure.cell.lengths()[0])
        delta[:, 1] -= initial_structure.cell.lengths()[1] * np.rint(delta[:, 1] / initial_structure.cell.lengths()[1])
        dists = np.sqrt((delta ** 2).sum(axis=1))

        # Adjust z position if too close to existing atoms
        while np.any(dists < min_dist):
            z += 0.1
            new_atom.positions[0][2] = z
            # Recalculate distances after adjusting z
            delta = all_positions - new_atom.positions[0]
            delta[:, 0] -= initial_structure.cell.lengths()[0] * np.rint(delta[:, 0] / initial_structure.cell.lengths()[0])
            delta[:, 1] -= initial_structure.cell.lengths()[1] * np.rint(delta[:, 1] / initial_structure.cell.lengths()[1])
            dists = np.sqrt((delta ** 2).sum(axis=1))
            if z > local_max_z:
                break

        if np.all(dists >= min_dist) and z <= local_max_z:
            initial_structure += new_atom
            placed = True


# Write the current structure to a data file in the batch directory
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
lmp.command("mass 2 195.084")   # Platinum

# Define Groups
lmp.command("group graphene type 1")
lmp.command("group platinum type 2")

# Fix the Momentum of the Graphene Sheet
lmp.command("fix fix_graphene_momentum graphene momentum 1 linear 1 1 1")

# Apply a reflecting wall at some z value for Pt atoms
lmp.command("fix upper_wall platinum wall/reflect zhi 40.0")

# NNP potential
lmp.command("pair_style allegro")
lmp.command("pair_coeff * * /home/common/akram/graphene_pt/allegro/20/results/run/deployed_model.pth C Pt")

# Define computes
lmp.command("compute atomicenergies all pe/atom")
lmp.command("compute totalatomicenergy all reduce sum c_atomicenergies")

# Thermodynamics
lmp.command("thermo 50")
lmp.command("thermo_style custom step temp vol press pe ke etotal c_totalatomicenergy")
lmp.command("thermo_modify flush yes")

# Define dump with per-atom energies
lmp.command("dump 1 all custom 50 traj.lmp id type x y z fx fy fz c_atomicenergies")

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
