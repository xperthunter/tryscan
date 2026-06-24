#!/usr/bin/python3

from itertools import product
import math
import sys

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
import numpy as np

from msalib import MSA
from dca import DCA

DEVMODE = True

class ALN(DCA):
	
	def __init__(self, lines, pdb):
		
		self.seqs = []
		self.resindices = dict()
		
		for i, line in enumerate(lines):
			self.seqs.append(line)
			self.resindices[i] = dict()
			for j, sym in enumerate(line):
				
				self.resindices[i][j] = j
		
		
		#self.seqs = self.seqs[:100]
		self.depth = len(self.seqs)
		self.length = len(lines[0])
		
		self.lens = [self.length] * self.depth
		
		q = [
			'A','C','D','E','F',
			'G','H','I','K','L',
			'M','N','P','Q','R',
			'S','T','V','W','Y',
			'.'
		]
		self.q  = q
		self.q2 = list(product(q, q))
		
		# initialize object attributes
		self.neff = None
		self.ma = None
		self.fi = None
		self.fij = None
		self.similarity_cutoff = None
		self.psuedo = None
		self.rescale = None
		self.mij = None
		self.pdb = pdb
		
		self._pcs = None
	
	
	def _test_contact(self, msa_index, protein_index):
		id1, id2 = msa_index
		res1, res2 = protein_index
		
		res1 += 1
		res2 += 1
		
		
		for atom1 in self.pdb[0][" "][res1].get_atoms():
			for atom2 in self.pdb[0][" "][res2].get_atoms():
				dis = atom1 - atom2
				if dis < self.cutoff:
					return True
		#sys.exit()
		return False
	
	
	
	def score_dij():
		pass
	
	def score_mij(self, cutoff=8.0):
		"""
		Score agreement between mututal information and structural contacts
		"""
		
		self.cutoff = cutoff
		
		
		scores = {}
		cumulative_scores = {}
		measures = 0
		for rank, (k,v) in enumerate(sorted(self.mij.items(), key = lambda x: x[1], reverse=True)):
			l = k
			#print(k)
			contact = self._test_contact(k, l)
			if contact is not None:
				if contact:
					scores[rank] = 1
				else:
					scores[rank] = 0
			
			if rank == 0:
				cumulative_scores[rank] = scores[rank]
			else:
				cumulative_scores[rank] = scores[rank] + cumulative_scores[rank-1]
		
		self.scores = scores
		self.cumulative_scores = cumulative_scores
		return
	
	def score_psicov(self, cons, cutoff=8.0):
		
		scores = {}
		cumulative_scores = {}
		measures = 0
		for rank, con in enumerate(cons):
			
			info = con.split()
			i = int(info[0]) - 1
			j = int(info[1]) - 1
			
			contact = self._test_contact((i,j), (i,j))
			if contact is not None:
				if contact:
					scores[rank] = 1
				else:
					scores[rank] = 0
			
			if rank == 0:
				cumulative_scores[rank] = scores[rank]
			else:
				cumulative_scores[rank] = scores[rank] + cumulative_scores[rank-1]
		
		self.psicov_scores = scores
		self.psicov_cums = cumulative_scores
		return
		
		
	
	

if __name__ == '__main__':
	
	import argparse
	import os
	
	
	parser = argparse.ArgumentParser(description="measure mutual information with new psuedo counts")

	parser.add_argument('--psicov', '-p', required=True, type=str, metavar='<str>', help="path to psicov data directory")
	parser.add_argument('--sim', '-s', default=0.8, type=float, metavar='<float>', help="max sequence identity for sequence clustering")
	parser.add_argument('--psuedo', '-l', default=1.0, type=float, metavar='<float>', help="pseudo count scale. psuedo counts added as l * n_eff")
	parser.add_argument('--rescale', '-r', action='store_true', help="rescale psuedo counts (optional)")
	parser.add_argument('--cutoff', '-c', default=8.0, type=float, metavar='<float>', help="distance cutoff to call contacts")
	
	args = parser.parse_args()
	
	ddir = os.path.abspath(args.psicov)
	assert(os.path.isdir(ddir))
	
	pdbs = os.path.join(ddir, "pdb")
	msas = os.path.join(ddir, "aln")
	seqs = os.path.join(ddir, "seq")
	cons = os.path.join(ddir, "con")
	
	
	assert(os.path.isdir(pdbs))
	assert(os.path.isdir(msas))
	assert(os.path.isdir(seqs))
	
	for aln in os.listdir(msas):
		if not os.path.isfile(os.path.join(msas,aln)): continue
		assert(aln.endswith(".aln"))
		
		pdbid = aln.split(".aln")[0]
		
		print(pdbid)
		print()
		
		with open(os.path.join(msas, aln), 'r') as fp:
			lines = [line.rstrip() for line in fp]
		
		pdb = os.path.join(pdbs, pdbid+'.pdb')
		assert(os.path.isfile(pdb))
		
		pdbser = PDBParser(PERMISSIVE=True, QUIET=True)
		structure = pdbser.get_structure(pdbid, pdb)
		
		comps = ALN(lines, structure)
		
		comps.measure_mij(similarity_cutoff=args.sim, psuedo=args.psuedo, rescale=True)
		
		comps.score_mij(cutoff=args.cutoff)
		
		print("with rescaling")
		for k,v in comps.cumulative_scores.items():
			if (k+1) % 10 != 0: continue
			
			print(f"rank {k+1:>4} correct: {v:>4}   acc: {v / (k+1):6.4f}")
			
			if (k+1) == 100: break
	
		print()
		
		comps.measure_mij(similarity_cutoff=args.sim, psuedo=args.psuedo, rescale=False)
		
		comps.score_mij(cutoff=args.cutoff)
		
		print("withoutrescaling")
		for k,v in comps.cumulative_scores.items():
			if (k+1) % 10 != 0: continue
			
			print(f"rank {k+1:>4} correct: {v:>4}   acc: {v / (k+1):6.4f}")
			
			if (k+1) == 100: break
	
		print()
		
		
		con = os.path.join(cons, pdbid+".out")
		assert(os.path.isfile(con))
		
		
		psicov_results = []
		with open(con, 'r') as fp:
			psicov_results = [line.rstrip() for line in fp]
		
		comps.score_psicov(psicov_results, cutoff=args.cutoff)
		
		for k,v in comps.psicov_cums.items():
			if (k+1) % 10 != 0: continue
			
			print(f"rank {k+1:>4} correct: {v:>4}   acc: {v / (k+1):6.4f}")
			
			if (k+1) == 100: break
		
		print()
		#sys.exit()
	
	
	
	
		
		
	
	
	
		
		
		