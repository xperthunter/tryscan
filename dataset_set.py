#!/usr/bin/python3

import argparse
from itertools import product
import json
import math
import os
import sys

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
import cppyy
import numpy as np
import scipy.stats

from msalib import MSA
from msalib import read_stockholm


parser = argparse.ArgumentParser(description="measure mutual information with new psuedo counts")

parser.add_argument('--msa', '-m', required=True, type=str, metavar='<str>', help="path to a msa in stockholm format")
parser.add_argument('--data', '-d', required=True, type=str, metavar='<str>', help="path to json file containing small dataset")
parser.add_argument('--pdbs', '-p', required=True, type=str, metavar='<str>', help="path to folder with tester structures")
parser.add_argument('--sim', '-s', default=0.8, type=float, metavar='<float>', help="max sequence identity for sequence clustering")
parser.add_argument('--psuedo', '-l', default=1.0, type=float, metavar='<float>', help="pseudo count scale. psuedo counts added as l * n_eff")
parser.add_argument('--rescale', '-r', action='store_true', help="rescale psuedo counts (optional)")
parser.add_argument('--cutoff', '-c', default=8.0, type=float, metavar='<float>', help="distance cutoff to call contacts")

args = parser.parse_args()
pdb_path = os.path.abspath(args.pdbs)

cases = {}
with open(args.data, 'r') as fp:
	cases = json.load(fp)

for msa in read_stockholm(args.msa):
	#print(msa.identifier)
	if msa.identifier not in cases: continue
	
	pdb = os.path.join(pdb_path, cases[msa.identifier]["pdb_test"])
	
	print()
	
	print(msa.identifier)
	print(msa.accession)
	print(f"num of seqs: {len(msa.seqs)}")
	print(f"msa width: {len(msa.seqs[0])}")
	
	pdbser = PDBParser(PERMISSIVE=1)
	structure = pdbser.get_structure(cases[msa.identifier]["member_id"], pdb)
	
	msa.measure_mij(similarity_cutoff=args.sim, psuedo=args.psuedo, rescale=True)
	
	print(f"neff: {msa.neff:6.2f}")
	
	msa.score_mij(test_id=cases[msa.identifier]["member_id"], pdb=structure, cutoff=args.cutoff)
	with_scaling = msa.cumulative_scores
	
	msa.measure_mij(similarity_cutoff=args.sim, psuedo=args.psuedo, rescale=False)
	msa.score_mij(test_id=cases[msa.identifier]["member_id"], pdb=structure, cutoff=args.cutoff)
	wout_scaling = msa.cumulative_scores
	
	
	for kr,ko in zip(with_scaling.keys(), wout_scaling.keys()):
		if (kr+1) % 10 != 0: continue
		
		vr = with_scaling[kr]
		vo = wout_scaling[ko]
		
		print(f"rank {kr+1:>4} correct -> {vr:>4} (scaling) {vo:>4} (w/o) acc: {vr / (kr+1):6.4f} (scaling) {vo / (kr+1):6.4f} (out)")
		
		if (kr+1) == 100: break
	
	print()
	