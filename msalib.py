#!/usr/bin/python3

import gzip
from itertools import product
import json
import math
import sys

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
import cppyy
import numpy as np
import scipy.stats


DEVMODE = True


AA = { # Amino Acid Frequencies (from Pfam 38.1)
	'A': 0.0846,
	'C': 0.0140,
	'D': 0.0563,
	'E': 0.0647,
	'F': 0.0421,
	'G': 0.0680,
	'H': 0.0221,
	'I': 0.0585,
	'K': 0.0527,
	'L': 0.1022,
	'M': 0.0209,
	'N': 0.0393,
	'P': 0.0442,
	'Q': 0.0377,
	'R': 0.0569,
	'S': 0.0647,
	'T': 0.0541,
	'V': 0.0702,
	'W': 0.0142,
	'Y': 0.0325,
}


B62 = { # BLOSUM62 scoring matrix
	'A':{'A':4,'R':-1,'N':-2,'D':-2,'C':0,'Q':-1,'E':-1,'G':0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':0,'W':-3,'Y':-2,'V':0},
	'R':{'A':-1,'R':5,'N':0,'D':-2,'C':-3,'Q':1,'E':0,'G':-2,'H':0,'I':-3,'L':-2,'K':2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
	'N':{'A':-2,'R':0,'N':6,'D':1,'C':-3,'Q':0,'E':0,'G':0,'H':1,'I':-3,'L':-3,'K':0,'M':-2,'F':-3,'P':-2,'S':1,'T':0,'W':-4,'Y':-2,'V':-3},
	'D':{'A':-2,'R':-2,'N':1,'D':6,'C':-3,'Q':0,'E':2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S':0,'T':-1,'W':-4,'Y':-3,'V':-3},
	'C':{'A':0,'R':-3,'N':-3,'D':-3,'C':9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
	'Q':{'A':-1,'R':1,'N':0,'D':0,'C':-3,'Q':5,'E':2,'G':-2,'H':0,'I':-3,'L':-2,'K':1,'M':0,'F':-3,'P':-1,'S':0,'T':-1,'W':-2,'Y':-1,'V':-2},
	'E':{'A':-1,'R':0,'N':0,'D':2,'C':-4,'Q':2,'E':5,'G':-2,'H':0,'I':-3,'L':-3,'K':1,'M':-2,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
	'G':{'A':0,'R':-2,'N':0,'D':-1,'C':-3,'Q':-2,'E':-2,'G':6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S':0,'T':-2,'W':-2,'Y':-3,'V':-3},
	'H':{'A':-2,'R':0,'N':1,'D':-1,'C':-3,'Q':0,'E':0,'G':-2,'H':8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y':2,'V':-3},
	'I':{'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I':4,'L':2,'K':-3,'M':1,'F':0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V':3},
	'L':{'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I':2,'L':4,'K':-2,'M':2,'F':0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V':1},
	'K':{'A':-1,'R':2,'N':0,'D':-1,'C':-3,'Q':1,'E':1,'G':-2,'H':-1,'I':-3,'L':-2,'K':5,'M':-1,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
	'M':{'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q':0,'E':-2,'G':-3,'H':-2,'I':1,'L':2,'K':-1,'M':5,'F':0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V':1},
	'F':{'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I':0,'L':0,'K':-3,'M':0,'F':6,'P':-4,'S':-2,'T':-2,'W':1,'Y':3,'V':-1},
	'P':{'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P':7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
	'S':{'A':1,'R':-1,'N':1,'D':0,'C':-1,'Q':0,'E':0,'G':0,'H':-1,'I':-2,'L':-2,'K':0,'M':-1,'F':-2,'P':-1,'S':4,'T':1,'W':-3,'Y':-2,'V':-2},
	'T':{'A':0,'R':-1,'N':0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':5,'W':-2,'Y':-2,'V':0},
	'W':{'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F':1,'P':-4,'S':-3,'T':-2,'W':11,'Y':2,'V':-3},
	'Y':{'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H':2,'I':-1,'L':-1,'K':-2,'M':-1,'F':3,'P':-3,'S':-2,'T':-2,'W':2,'Y':7,'V':-1},
	'V':{'A':0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I':3,'L':1,'K':-2,'M':1,'F':-1,'P':-2,'S':-2,'T':0,'W':-3,'Y':-1,'V':4},
}


def get_fp(filename):
	"""Returns a file pointer for reading based on file name"""
	if   filename.endswith('.gz'): return gzip.open(filename, 'rt')
	elif filename == '-':          return sys.stdin
	else:                          return open(filename)


def read_fasta(filename):
	"""Simple fasta file iterator: yields defline, seq"""
	name = None
	seqs = []
	fp = get_fp(filename)
	while True:
		line = fp.readline()
		if line == '': break
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield name, seq
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	yield name, ''.join(seqs)
	fp.close()


def read_stockholm(filename):
	"""Stockholm file iterator: yields msa objects"""
	fp = get_fp(filename)
	lines = []
	for line in fp:
		if line.startswith('//'):
			yield MSA(lines)
			lines = []
		else:
			lines.append(line.rstrip())
	fp.close()


#############
## C area ###
#############

cppyy.cppdef("""
extern "C" {
#include <string.h>
#include <stdio.h>

void get_ma(char **seqs, int *lens, int size, float max_similarity, int *results) {
	int max_mismatch = 0;
	for (int i = 0; i < size; i++) {
		char *s1 = seqs[i];
		int slen = strlen(s1);
		results[i] = 1;
		max_mismatch = (int) (lens[i] * (1 - max_similarity) + 1.0);
		for (int j = 0; j < size; j++) {
			if (i == j) continue;
			char *s2 = seqs[j];
			int mismatch = 0;
			for (int k = 0; k < slen; k++){
				if (s1[k] == '.') continue;
				if (s1[k] == '-') continue;
				if (s2[k] == '.') continue;
				if (s2[k] == '-') continue;
				if (s1[k] != s2[k]) mismatch++;
				if (mismatch >= max_mismatch) break;
			}
			if (mismatch < max_mismatch) results[i] = results[i] + 1;
		}
	}
}}""")


class MSA:
	"""
	Reading MSAs from Pfam families and analyzing amino acid statistics and mutual information
	Contains method to score agreement between mutual information and representative structures
	from a a family.
	"""
	def __init__(self, lines, aln=None):
		"""
		Initialize a MSA object from an alignment in stockholm format
		"""
		
		self.identifier = None # release identifier
		self.accession = None  # stable identifier
		self.description = []  # to be joined later
		self.type = None       # Domain, Family, Repeat, Colied-Coil, Disorderd, Motif
		self.length = None
		self.depth = None
		self.uids = []         # short identifier
		self.lids = []         # long identifier (shows subsequence)
		self.sub_seqs = []
		self.seqs = []
		self.lens = []
		self.uid_index = {}
		self.lid_index = {}
		self.cons = None
		self.resindices = dict()

		if not lines[0].startswith('# STOCKHOLM 1.0'):
			sys.exit('MSA constructor error')

		for line in lines[1:]:
			if line.startswith('#=GF'):
				tag = line[5:7]
				val = line[10:]
				match tag:
					case 'ID': self.identifier = val
					case 'AC': self.accession = val
					case 'DE': self.description.append(val)
					case 'TP': self.type = val
			elif line.startswith('#=GS'):
				foo, lid, ac, uid = line.split()
				self.uids.append(uid)
				self.lids.append(lid)
				ll = lid.split('/')[-1].split('-')
				ll = [int(l) for l in ll]
				self.sub_seqs.append(tuple(ll))
				seqlen = ll[1] - ll[0]
				assert(seqlen > 0)
				self.lens.append(seqlen)
			elif line.startswith('#=GC'):
				self.cons = line.split()[2]
			elif line.startswith('#'):
				if line.startswith('#=GR'): pass
				else: sys.exit('unrecognized line')
			else:
				lid, seq = line.split()
				entry = ""
				for c in seq:
					if c == '.' or c == '-': continue
					elif not c.isupper(): continue
					else:
						entry += c
				
				if len(entry) == 0:
					self.seqs.append("")
				else:
					self.seqs.append(seq.upper())
		
		
		self.uids = [uid for k, uid in enumerate(self.uids) if len(self.seqs[k]) != 0]
		self.lids = [lid for k, lid in enumerate(self.lids) if len(self.seqs[k]) != 0]
		self.sub_seqs = [sub for k, sub in enumerate(self.sub_seqs) if len(self.seqs[k]) != 0]
		self.lens = [ll for k, ll in enumerate(self.lens) if len(self.seqs[k]) != 0]
		self.seqs = [seq for k, seq in enumerate(self.seqs) if len(self.seqs[k]) != 0]
		
		#self.seqs = [[s for i,elm in enumerate(s) if self.cons[i] != '.'] for s in self.seqs]
		
		self.description = ' '.join(self.description)
		self.length = len(self.seqs[0])
		self.depth = len(self.seqs)
		
		assert(len(self.cons) == self.length)
		
		cons_index = dict()
		counter = 0
		for i,elm in enumerate(self.cons):
			if elm == '.' or elm == '-': continue
			cons_index[i] = counter
			counter += 1
		
	#	print(json.dumps(cons_index,indent=2))
		#sys.exit()
		
		for k, (ends, lid, uid, seq) in enumerate(zip(self.sub_seqs, self.lids, self.uids, self.seqs)):
			
			self.uid_index[uid] = k
			self.lid_index[lid] = k

			beg, end = ends
			
			j = 0
			self.resindices[k] = dict()
			for i, sym in enumerate(seq):
				if sym.upper() not in AA: continue
				#if self.cons[i] == '.': continue
				
				j += 1
				if i in cons_index:
					self.resindices[k][cons_index[i]] = beg + j - 1
		
		self.seqs = [''.join([elm for i,elm in enumerate(s) if self.cons[i] != '.' and self.cons[i] != '-']) for s in self.seqs]
		#print(self.seqs[0])
		#print(len(self.seqs[0]))
		#print(len(list(cons_index.keys())))
		assert(len(self.seqs[0]) == len(list(cons_index.keys())))
		self.length = len(self.seqs[0])
		
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
		
		self._pcs = None
	
	
	def write(self, fp):
		"""
		Write out MSA in stockholm
		"""
		print('# STOCKHOLM 1.0', file=fp)
		print('#=GF ID', self.identifier, file=fp)
		print('#=GF AC', self.accession, file=fp)
		print('#=GF DE', self.description, file=fp)
		print('#=GF TP', self.type, file=fp)
		for lid, uid in zip(self.lids, self.uids):
			print('#=GS', lid, 'AC', uid, file=fp)
		for lid, seq in zip(self.lids, self.seqs):
			print(lid, seq, sep='\t', file=fp)
		print('//', file=fp)
	
	
	def column(self, n):
		"""
		MSA column getter
		"""
		letters = []
		for seq in self.seqs:
			letters.append(seq[n])
		return ''.join(letters)
	

	def _seq_cluster(self, sim=None):
		"""
		internal function to compute sequence clustering
		calling out to cppyy
		
		Sets
		----
		ma: `dict`
			key -> msa member index
			val -> number of msa members with sequence similarity > similarity_cutoff
		"""
		
		if self.similarity_cutoff is None:
			assert(sim is not None)
			cutoff = sim
		else:
			cutoff = self.similarity_cutoff
		
		results = np.zeros(self.depth, dtype=np.intc)
		lens = np.array(self.lens, dtype=np.intc)
		
		cppyy.gbl.get_ma(self.seqs, lens, self.depth, cutoff, results)
		
		self.ma = {k:v for k, v in enumerate(results)}
		
		return
		
		
	def set_neff(self):
		"""
		compute N_eff
		
		Sets
		----
		neff: `float`
			effective number of sequences
		"""
		
		neff = 0
		for v in self.ma.values(): neff += 1/v
		
		self.neff = neff
		
		return
	
	
	def _set_fi(self):
		"""
		compute the single column amino acid frequency distribution
		
		Sets
		----
		fi: `dict`
			key -> column index
			val -> `dict` of amino acid frequencies in that column
		
		Description
		-----------
		`set_fi` is computing amino acid frequency distribution for each separate MSA column.
		
		We define f_i(A) as the frequency of observing amino acid *A* in column *i*. This is equal to
		
		              1         (   ps    [    M        1                           ]) 
		f_i(A) = -----------  * (  ---- + [ sum__to   ----- * if(a(i) == A) ? 1 : 0 ])
		         n_eff  + ps    (   q     [   a=1      m^a                          ])
		
		- i     = column number
		- A     = amino acid letter
		- n_eff = effective number of sequences given sequence identity threshold
		- ps    = psuedo counts. psuedo counts are provided from the user as a scaling factor, self.psuedo > 0.
					the number of psuedo counts add is self.psuedo * n_eff
		- q     = 21, length of amino acid alphabet + 1 for gap
		- m^a   = number of sequences with sequence identity >= threshold to sequence `a`
		- a(i)  = amino acid at sequence a at column i
		"""
		
		assert(self.neff is not None)
		assert(self._pcs is not None)
		assert(self.ma   is not None)
		
		fi = dict()
		
		for i in range(self.length):
			#if self.cons[i] == '.': continue
			
			if self.rescale:
				fi[i] = dict()
			else:
				fi[i] = {aa: ((self._pcs) / len(self.q)) for aa in self.q}
				
			col = self.column(i)
			
			for n, elm in enumerate(col):
				if elm == '-': elm = '.';
				if elm.upper() not in self.q: continue
				if self.rescale:
					if elm not in fi[i]: fi[i][elm] = (self._pcs) / len(self.q)
				
				fi[i][elm] += 1 / self.ma[n]
			
			for k in fi[i].keys(): fi[i][k] = fi[i][k] / (self.neff + self._pcs)
		
		self.fi = fi
		return
	
	
	def _rescaled_counts(self):
		"""
		compute the rescaled psuedo counts
		
		Returns
		-------
		`float`: rescaled psuedo counts given current pseudo count value and neff
		
		Description
		-----------
		`_rescaled_counts` computes the necessary amount of psuedo counts needs such that:
		
		              f_ij(A,B)
		          --------------- = 1
		          f_i(A) * f_j(B)
			
			when there are no observations for amino acids A,B in columns i,j
			
		Given equations for f_i(A) and f_ij(A, B), the equation for rescaled psuedo counts is:
		
		                   self.psuedo ** 2
		rescaled_counts = -------------------  * n_eff
					      1 + 2 * self.psuedo
		
		Dervivation of this relation is provided in the README
		"""
		assert(self.neff is not None)
		assert(self._pcs is not None)
		assert(self.ma   is not None)
		
		return (self.psuedo**2 / (1.0 + (2.0*self.psuedo))) * self.neff
	
	
	def _set_fij(self):
		"""
		compute the pairwise column amino acid frequeny distribution
		
		Sets
		----
		fij: `dict`
			key -> `tuple` ordered pair column indices, non-redundant ordered pairs
			val -> `dict` of amino acid tuples frequencies
		
		Description
		-----------
		`set_fij` is computing amino acid frequency distribution for non-redundant MSA column pairs.
		
		We define f_ij(A,B) as the frequency of observing amino acid pair *A* and *B* in column *i* and column *j*.
		This is equal to:
		
		              1         (   ps    [    M        1                                         ]) 
		f_i(A) = -----------  * (  ---- + [ sum__to   ----- * if(a(i) == A and a(j) == B) ? 1 : 0 ])
		         n_eff  + ps    (  q**2   [   a=1      m^a                                        ])
		
		- i     = column number
		- A,B   = amino acid letter
		- n_eff = effective number of sequences given sequence identity threshold
		- ps    = psuedo counts. psuedo counts are provided from the user as a scaling factor, self.psuedo > 0.
					the number of psuedo counts added is self.psuedo * n_eff
		- q     = 21, length of amino acid alphabet + 1 for gap
		- m^a   = number of sequences with sequence identity >= threshold to sequence `a`
		- a(i)  = amino acid at sequence a at column i		
		"""
		
		assert(self.neff is not None)
		assert(self._pcs is not None)
		assert(self.ma   is not None)
		
		fij = dict()
		
		if self.rescale:
			rpcs = self._rescaled_counts()
		
		for i in range(self.length):
			#if self.cons[i] == '.': continue
			for j in range(i+5, self.length):
				#if self.cons[j] == '.': continue
				
				if self.rescale:
					fij[(i,j)] = dict()
				else:
					fij[(i,j)] = {pair: (self._pcs) / len(self.q2) for pair in self.q2}
				
				col_i = self.column(i)
				col_j = self.column(j)
				
				for n, (ei, ej) in enumerate(zip(col_i, col_j)):
					if ei == '-': ei = '.'
					if ej == '-': ej = '.'
					if ei not in self.q or ej not in self.q: continue
					if self.rescale:
						if (ei,ej) not in fij[(i,j)]:
							fij[(i,j)][(ei,ej)] = rpcs / len(self.q2)
					
					fij[(i,j)][(ei,ej)] += 1 / self.ma[n]
				
				for k in fij[(i,j)].keys():
					if self.rescale:
						fij[(i,j)][k] = fij[(i,j)][k] / (self.neff + rpcs)
					else:
						fij[(i,j)][k] = fij[(i,j)][k] / (self.neff + self._pcs)
		
		self.fij = fij
		return
	
	
	def _set_fijk(self):
		"""
		compute three point correlations
		"""

		for i in range(self.length):
			col_i = self.column(i)
			for j in range(i+1, self.length):
				col_j = self.column(j)
				for k in range(max(i+5, j+1), self.length):
					col_k = self.column(k)
					for n, (ei, ej, ek) in enumerate(zip(col_i, col_j, col_k)):
						if ei == '-': ei = '.'
						if ej == '-': ej = '.'
						if ek == '-': ek = '.'

						if ei not in self.q or ej not in self.q or ek not in self.q: continue

						if self.rescale:
							if (ei,ej,ek) not in fijk[(i,j,k)]




	def measure_mij(self, similarity_cutoff=0.8, psuedo=1.0, rescale=False):
		"""
		measure mutual information for pairs of columns
		
		Parameters
		----------
		similarity_cutoff: `float`, (0,1]
			maximum sequence identity for MSA sequence clustering
		
		psuedo: `float`, > 0.0
			parameter controlling how much psuedo counts to add
			this adds ``psuedo * N_eff `` psuedo counts to f_i and f_ij statistics
		
		rescale: `bool`
			two methods for adding psuedo counts
			* False : both f_i, f_ij receive same number of psuedo counts
			* True  : f_i and f_ij receive different number of psuedo counts
				`rescale` sets ratio of un-observed pairs and column marginals to 1.
				f_ij / (f_i * f_j) == 1 under rescale
				* see README for more information
		
		Returns
		-------
		- mij:	`dict`, keys cols (i,j) in MSA order, vals mutual information for columns	
		"""
		
		# consider a global variable to debug or not, instead of if true
		# do we want to use raise instead ??
		if DEVMODE:
			assert isinstance(similarity_cutoff, float), f"sequence similarity cutoff: {similarity_cutoff} invalid -- `float` required"
			assert similarity_cutoff > 0.0 and similarity_cutoff <= 1.0, f"sequence similarity cutoff: {similarity_cutoff} invalid -- needs to be (0,1]"
			assert isinstance(psuedo, float), f"input psuedo scale {psuedo} has unexpected type"
			assert psuedo > 0.0, f"psuedo count scale cannot be negative"
			assert isinstance(rescale, bool),f"argument `rescale` is of unexpected type"
		
		self.similarity_cutoff = similarity_cutoff
		self.psuedo = psuedo
		self.rescale = rescale
		
		self._seq_cluster()
		self.set_neff()
		
		self.psuedo = len(self.q) / self.neff
		
		self._pcs = self.psuedo * self.neff
		
		self._set_fi()
		self._set_fij()
		
		mij = dict()
		
		for (i,j) in self.fij.keys():
			
			info = 0
			if self.rescale:
				
				l_rescaled = self._rescaled_counts() / self.neff # only want the the rescaled factor
				psuedo_ij = l_rescaled / (len(self.q2) * (1 + l_rescaled))
				psuedo_i  = self.psuedo / (len(self.q) * (1+self.psuedo))
				for (ai,aj) in self.q2:
					if (ai,aj) not in self.fij[(i,j)]:
						if ai in self.fi[i] and aj not in self.fi[j]:
							info += psuedo_ij * math.log2(psuedo_ij / (self.fi[i][ai] * psuedo_i))
						elif ai not in self.fi[i] and aj in self.fi[j]:
							info += psuedo_ij * math.log2(psuedo_ij / psuedo_i * self.fi[j][aj])
						else:
							score = psuedo_ij / (psuedo_i ** 2)
							assert(math.isclose(score, 1.0))
					else:
						info += self.fij[(i,j)][(ai,aj)] * math.log2(self.fij[(i,j)][(ai,aj)] / (self.fi[i][ai] * self.fi[j][aj]))
			else:
				for (ai, aj) in self.q2:
					info += self.fij[(i,j)][(ai,aj)] * math.log2(self.fij[(i,j)][(ai,aj)] / (self.fi[i][ai] * self.fi[j][aj]))
			
			mij[(i,j)] = info
		
		self.mij = mij
		return
	
	
	def _test_contact(self, msa_index, protein_index):
		id1, id2 = msa_index
		res1, res2 = protein_index
		
		aa1 = self.seqs[self.test_index][id1]
		aa2 = self.seqs[self.test_index][id2]
		
		if aa1.upper() not in self.q and aa2.upper() not in self.q:
			return None
		
		assert(aa1 == three_to_one[self.pdb[0]["A"][res1].get_resname()])
		assert(aa2 == three_to_one[self.pdb[0]["A"][res2].get_resname()])
		
		for atom1 in self.pdb[0]["A"][res1].get_atoms():
			for atom2 in self.pdb[0]["A"][res2].get_atoms():
				dis = atom1 - atom2
				if dis < self.cutoff:
					return True
		
		return False
	
	
	def score_mij(self, test_id=None, pdb=None, cutoff=8.0):
		"""
		Score agreement between mututal information and structural contacts
		
		Parameters
		----------
		test_id: `str` id of entry in MSA to base scoring off of
		pdb: BioPython PDB structure object to use for scoring
		cutoff: `float` distance cutoff for calling contacts
	
		Returns
		-------
		"""
		
		if DEVMODE:
			assert test_id in self.uid_index, f"test_id `{test_id}` not found in MSA"
			assert isinstance(pdb, object), f"unexpected type {type(pdb)} for pdb argument"
			assert isinstance(cutoff, float), f"unexpected type {type(cutoff)} for distance cutoff"
		
		
		self.pdb = pdb
		self.cutoff = cutoff
		self.test_id = test_id
		self.test_index = self.uid_index[test_id]
		
		
		scores = {}
		cumulative_scores = {}
		measures = 0
		#print(f"test_id {self.test_id}")
		#print(f"test index {self.test_index}")
		#print(json.dumps(self.resindices[self.test_index],indent=2))
		skips = 0
		for rank, (k,v) in enumerate(sorted(self.mij.items(), key = lambda x: x[1], reverse=True)):
			try:
				l = (
					self.resindices[self.test_index][int(k[0])],
					self.resindices[self.test_index][int(k[1])]
				)
			except:
				skips += 1
				continue
			
			contact = self._test_contact(k, l)
			if contact is not None:
				if contact:
					scores[rank-skips] = 1
				else:
					scores[rank-skips] = 0
			
			if rank-skips == 0:
				cumulative_scores[rank-skips] = scores[rank-skips]
			else:
				cumulative_scores[rank-skips] = scores[rank-skips] + cumulative_scores[rank-skips-1]
		
		self.scores = scores
		self.cumulative_scores = cumulative_scores
		return


if __name__ == '__main__':
	pass
	sys.exit()
