#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7


import sys
sys.path.append("/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta")
sys.path.append("/opt/cesga/easybuild-cesga/software/MPI/gcc/6.4.0/openmpi/2.1.1/mpi4py/3.0.2-python-3.7.0/lib/python3.7/site-packages")

import pyrosetta
import pyrosetta.rosetta as rosetta

from mpi_rosetta import *

from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor,
)

from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.scoring import CA_rmsd


class StructureReader:
    def pose_structure(self, pose):
        """
        Extracts and displays various structural properties of the input  <pose>
        and its  <display_residues>  including:
            -PDB numbering
            -chain identification
            -sequence
            -secondary structure
        
        """

        display_residues = range(1, pose.total_residue() + 1)
        # store the pose's number of residues, example Python syntax
        nres = pose.total_residue()
        # 1. obtain the pose's sequence
        sequence = pose.sequence()
        
        # 2. obtain a list of PDB numbering and icode as a single string
        pdb_info = pose.pdb_info()
        PDB_nums = [(str( pdb_info.number(i)) + pdb_info.icode(i)).strip()
            for i in range(1, nres + 1)]
        # 3. obtains a list of the chains organized by residue
        chains = [pdb_info.chain(i) for i in range(1, nres + 1)]
        # 4. extracts a list of the unique chain IDs
        unique_chains = []
        for c in chains:
            if c not in unique_chains:
                unique_chains.append(c)
            # start outputting information to screen
        print('\n' + '='*80)
        #print('Loaded from' , pdb_info.name())
        print(nres , 'residues')
        print(len(unique_chains), 'chain(s) ('+ str(unique_chains)[1:-1] + ')')
        print('Sequence:\n' + sequence)
        
        # this object is contained in PyRosetta v2.0 and above
        # 5. obtain the pose's secondary structure as predicted by PyRosetta's
        #    built-in DSSP algorithm
        DSSP = protocols.moves.DsspMover()
        DSSP.apply(pose)    # populates the pose's Pose.secstruct
        ss = pose.secstruct()
        print( 'Secondary Structure:\n' + ss )
        print( '\t' + str(100. * ss.count('H') / len(ss))[:4] + '% Helical' )
        print( '\t' + str(100. * ss.count('E') / len(ss))[:4] + '% Sheet' )
        print( '\t' + str(100. * ss.count('L') / len(ss))[:4] + '% Loop' )
        
        # 6. obtain the phi, psi, and omega torsion angles
        phis = [pose.phi(i) for i in range(1, nres + 1)]
        psis = [pose.psi(i) for i in range(1, nres + 1)]
        omegas = [pose.omega(i) for i in range(1, nres + 1)]
        
        # this object is contained in PyRosetta v2.0 and above
        # create a PyMOLMover for exporting structures directly to PyMOL
        #pymover = PyMOLMover()
        #pymover.apply(pose)    # export the structure to PyMOL (optional)
        self.display_structure(pose, sequence, chains, phis, psis, omegas, ss)
            
    def display_structure(self, pose, sequence, chains, phis, psis, omegas, ss):
        # 7. output information on the requested residues
        # use a simple dictionary to make output nicer
        ss_dict = {'L':'Loop', 'H':'Helix', 'E':'Strand'}
        display_residues = list(range(1, pose.total_residue() + 1))
        for i in display_residues:
            print( '='*80 )
            print( 'Pose numbered Residue', i )
            #print( 'PDB numbered Residue', PDB_nums[i-1] )
            print( 'Single Letter:', sequence[i-1] )
            print( 'Chain:', chains[i-1] )
            print( 'Secondary Structure:', ss_dict[ss[i-1]] )
            print( 'Phi:', phis[i-1] )
            print( 'Psi:', psis[i-1] )
            print( 'Omega:', omegas[i-1] )
            # extract the chis
            #chis = [pose.chi(j + 1, i) for j in range(pose.residue(i).nchi() )]
            #for chi_no in range(len(chis)):
            #    print( 'Chi ' + str(chi_no + 1) + ':', chis[chi_no] )
            print( '='*80 )

    def get_from_file(self, pdb_filename):
        pose = Pose()
        pose_from_file(pose, pdb_filename)
        to_centroid = SwitchResidueTypeSetMover('centroid')
        to_centroid.apply(pose)
        nres = pose.total_residue()
        #for i in range(1, nres + 1):
        #   pose.set_omega(i, 180)
        return pose

class StraightMover():
    def __init__(self, pose):
        straight = Pose()
        straight.assign(pose)
        nres = pose.total_residue()
        for i in range(1, nres + 1):
            straight.set_phi(i, 0)
            straight.set_psi(i, 0)
        self.straight_model = Pose()
        self.native = Pose()
        self.straight_model.assign(straight)
        self.native.assign(pose)
        
    def get_model(self):
        straight = Pose()
        straight.assign(self.straight_model)
        return straight

