
import numpy as np
from neuralnet import *

import os
import pyrosetta
import pyrosetta.rosetta as rosetta


from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor,
)

from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.scoring import CA_rmsd

from structural_alignment import kabsch_alignment



total_residue = pose1.total_residue()

        kabsch_alignment(pose1, pose2, range(1, total_residue + 1), range(1, total_residue + 1))
        # RMSD calculated by my own function
        for i in range(1, total_residue + 1):
            calculateRMS(pose1, pose2, i, output, show_index)

def calculateRMS(pose1, pose2, i, total_square)
    v1 = pose1.residue(i).xyz("CA")
    v2 = pose2.residue(i).xyz("CA")
    square = (v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2
    total_square = total_square + square
    root = math.sqrt(square)
    return total_square

    if index:
        output.write(str(i) + '\t' + str(root) + '\n')
    else:
        output.write(str(root) + '\n')
