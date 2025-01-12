import os
import sys
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell
from ovito.io import import_file
from ovito.modifiers import ClusterAnalysisModifier, SelectTypeModifier


# Parameters
N_H2 = 500               # Number of H2 molecules 
min_dist = 4.0           # Minimum initial distance in Å between new and base H2 molecules
initial_z_offset = 5.0   # Initial z-height for the first Pt atoms above graphene
max_z_offset = 20.0      # Maximum z-height offset from graphene


combined_structure = read('../final_atoms.extxyz')

# Modify the c-axis length if needed
cell = combined_structure.get_cell()
cell[2, 2] = 100  # Setting the c-axis length
combined_structure.set_cell(cell, scale_atoms=False)


#####################################
## Add H2 molecules
#####################################
cell_lengths = combined_structure.cell.lengths()[:2]

max_z_pt_c = np.max(combined_structure.positions[:,2])

# Generate H2 molecules on top 
for _ in range(N_H2):
    placed = False
    while not placed:
        # Generate a random seed 
        random_xy = np.random.rand(2)

        # Randomly select x, y within the supercell boundaries
        x, y = random_xy * cell_lengths

        # Start placing an H2 molecule at a safe height above the max-z atom near that position
        z = max_z_pt_c + initial_z_offset  # Start at initial z
        max_z = max_z_pt_c + max_z_offset  # max z

        # H2 bond length is approximately 0.74 Å
        bond_length = 0.74999

        # Positions of the two H atoms in the H2 molecule
        positions = [[x, y, z], [x, y, z + bond_length]]

        new_atoms = Atoms('H2', positions=positions)

        # Check distances from the new H2 atoms to all existing atoms in the structure
        all_positions = combined_structure.positions

        # For each atom in new_atoms, compute distances to all existing atoms
        dists1 = np.sqrt(np.sum((all_positions - new_atoms.positions[0])**2, axis=1))
        dists2 = np.sqrt(np.sum((all_positions - new_atoms.positions[1])**2, axis=1))

        # Determine the lowest possible z position maintaining the minimum distance
        while np.any(dists1 < min_dist) or np.any(dists2 < min_dist):
            z += 0.5  # Increase z slightly
            if z >= max_z:
                break  # Cannot place this molecule, try a new position
            new_atoms.positions[0][2] = z
            new_atoms.positions[1][2] = z + bond_length
            dists1 = np.sqrt(np.sum((all_positions - new_atoms.positions[0])**2, axis=1))
            dists2 = np.sqrt(np.sum((all_positions - new_atoms.positions[1])**2, axis=1))

        # If minimum distance condition is satisfied, add the atoms to the structure
        if (np.all(dists1 >= min_dist) and np.all(dists2 >= min_dist)) and (z < max_z):
            combined_structure += new_atoms
            placed = True

# Set the initial_structure
initial_structure = combined_structure.copy()

# Save as extxyz
write('initial_structure.extxyz', initial_structure)
#####################################
## Identify Pt clusters
#####################################
# Load your initial structure
pipeline = import_file('initial_structure.extxyz')

# Select Pt atoms (adjust type if needed)
pipeline.modifiers.append(SelectTypeModifier(types={'Pt'}))

# Perform cluster analysis on selected atoms
cluster_modifier = ClusterAnalysisModifier(cutoff=3.0)  # Adjust cutoff as needed
cluster_modifier.only_selected = True
pipeline.modifiers.append(cluster_modifier)

# Compute data
data = pipeline.compute()

# Access the cluster IDs for all particles
cluster_ids = data.particles['Cluster']

# Use numpy array indices as atom IDs, filtered by selected atoms
atom_ids = np.arange(len(cluster_ids))  # This creates an array of indices [0, 1, 2, ..., N-1]

# Group the atom IDs by cluster, neglecting cluster 0
clusters = {}

# Use a boolean mask to filter out particles not in clusters (cluster 0)
nonzero_cluster_mask = cluster_ids > 0
filtered_cluster_ids = cluster_ids[nonzero_cluster_mask]
filtered_atom_ids = atom_ids[nonzero_cluster_mask]

# Group atom IDs by cluster ID
unique_clusters = set(filtered_cluster_ids)
for cluster_id in unique_clusters:
    # Get all atoms belonging to the current cluster_id
    cluster_atoms = filtered_atom_ids[filtered_cluster_ids == cluster_id]
    clusters[cluster_id] = cluster_atoms

# Write atom IDs to temporary files
for cluster_id, atom_ids in clusters.items():
    # Write atom IDs to a temporary file
    filename = f"cluster_{cluster_id}.txt"
    with open(filename, 'w') as f:
        for atom_id in atom_ids:
            f.write(f"{int(atom_id)}\n")