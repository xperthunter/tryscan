#!/usr/bin/python3

from itertools import product
import math
import sys

import numpy as np

from msalib import MSA

DEVMODE = True

class DCA(MSA):
	
	def __init__(self, lines):
		super().__init__(lines)
		
		self.q_dca = [a for a in self.q if a != '.']
		assert(len(self.q_dca) + 1 == len(self.q))
		self.q2_dca = list(product(self.q_dca, self.q_dca))
		
		self.aa_map = {a: i for i, a in enumerate(self.q)}
	
	
	def compute_fi(self):
		
		pfreq = (self.psuedo * self.neff) / len(self.q)
		
		fi = np.full((self.length, len(self.q)), pfreq)
		
		for i in range(self.length):
			col = self.column(i)
			
			for n,elm in enumerate(col):
				if elm == '-': elm = '.';
				if elm.upper() not in self.q: continue
				fi[i, self.aa_map[elm]] += 1 / self.ma[n]
		
		fi /= (1 + self.psuedo) * self.neff
		
		self.fi = fi
		return
	
	
	def compute_fij(self):
		
		pfreqs = (self.psuedo * self.neff) / len(self.q2)
		
		fij = np.full((self.length, self.length, len(self.q), len(self.q)), pfreqs)
		
		for i in range(self.length):
			ci = self.column(i)
			for j in range(self.length):
				cj = self.column(j)
				
				for n, (a, b) in enumerate(zip(ci, cj)):
					if a == '-': a = '.'
					if b == '-': b = '.'
					
					if a.upper() not in self.q: continue
					if b.upper() not in self.q: continue
					
					A = self.aa_map[a]
					B = self.aa_map[b]
					
					fij[i, j, A, B] += 1 / self.ma[n]
		
		fij /= (1+ self.psuedo) * self.neff
		
		self.fij = fij
		return
	
	
	def compute_cov(self):
		q_red = len(self.q) - 1
		dim = self.length * q_red
		
		cov = np.zeros((dim, dim))
		
		def idx(ii, aa):
			return ii * q_red + aa
		
		for i in range(self.length):
			for j in range(self.length):
				for a in range(q_red):
					for b in range(q_red):
						cov[idx(i,a), idx(j,b)] = self.fij[i, j, a, b] - (self.fi[i, a] * self.fi[j, b])
		
		self.cov = cov
	
	
	def infer_couplings(self):
		C_inv = np.linalg.pinv(self.cov)
		
		eij = -C_inv
		self.eij = eij
		return
	
	
	def tilde_fields(self, f_i, f_j, J_ij):
		
		dim = len(self.q)
		ui = np.full(dim, 1.0/dim)
		uj = np.full(dim, 1.0/dim)
		
		for _ in range(self.max_iter):
			
			ui_prev = ui.copy()
			uj_prev = uj.copy()
			
			
			tmp_i = np.dot(J_ij, uj)
			tmp_j = np.dot(J_ij.T, ui)
			
			ui = f_i / tmp_i
			ui = ui / np.sum(ui)
			
			uj = f_j / tmp_j
			uj = uj / np.sum(uj)
			
			diff = max(
				np.absolute(ui - ui_prev).max(),
				np.absolute(uj - uj_prev).max()
			)
			
			if diff < self.tol: break
		
		return ui, uj
	
	
	def get_block(self, i, j):
		
		q = len(self.q)
		block = np.zeros((q, q))
		
		def idx(ii, aa):
			return ii * (q-1) + aa
		
		for a in range(q-1):
			for b in range(q-1):
				block[a, b] = self.eij[idx(i, a), idx(j, b)]
		
		return block
	
	
	def direct_information(self):
		
		L = self.length
		q = len(self.q)
		#psuedo_i  = self.psuedo / (self.neff * (1 + self.psuedo))
		
		dij = dict()
		
		for i in range(L):
			for j in range(i+1, L):
				
				# extract couplings
				e_ij = self.get_block(i, j)
				
				# exponentiate
				J_ij = np.exp(e_ij)
				
				# get marginals
				f_i = self.fi[i,:]
				f_j = self.fi[j,:]
				
				# get tilde fields
				ui, uj = self.tilde_fields(
					f_i,
					f_j,
					J_ij
				)
				
				ui = ui.flatten()
				uj = uj.flatten()
				
				# build P_dir
				P_dir = np.zeros((q, q))
				for a in range(q):
					for b in range(q):
						P_dir[a, b] = J_ij[a,b] * ui[a] * uj[b]
				
				# normalize
				P_dir /= P_dir.sum()
				
				# independent model
				P_ind = np.outer(f_i, f_j)
				
				# compute DI
				eps = 1e-12
				ratio = (P_dir + eps) / (P_ind + eps)
				DI_val = np.sum(P_dir * np.log2(ratio))
				
				dij[(i,j)] = DI_val
				
		self.dij = dij
		return
	
	
	def fit(self, similarity_cutoff=0.8, psuedo=1.0, max_iter=1000, tol=1e-4):
		
		self.similarity_cutoff = similarity_cutoff
		self.psuedo = psuedo
		self.max_iter = max_iter
		self.tol = tol
		
		self._seq_cluster()
		self.set_neff()
		
		self.compute_fi()
		self.compute_fij()
		
		self.compute_cov()
		self.infer_couplings()
		
		self.direct_information()
		return
	
	
	def score_dij(self, test_id=None, pdb=None, cutoff=8.0):
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
		for rank, (k,v) in enumerate(sorted(self.dij.items(), key = lambda x: x[1], reverse=True)):
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
		

if __name__ == "__main__":
	pass


"""
def measure_cij(self):
		
		cij = {}
		if self.rescale:
			l_rescaled = self._rescaled_counts() / self.neff # only want the the rescaled factor
			psuedo_ij = l_rescaled / (len(self.q2) * (1 + l_rescaled))
			psuedo_i  = self.psuedo / (len(self.q) * (1 + self.psuedo))
			for (i,j) in self.fij.keys():
				cij[(i,j)] = dict()
				for (ai,aj) in self.q2_dca:
					if (ai,aj) not in self.fij[(i,j)]:
						if ai in self.fi[i] and aj not in self.fi[j]:
							cij[(i,j)][(ai,aj)] = psuedo_ij - (self.fi[i][ai] * psuedo_i)
						elif ai not in self.fi[i] and aj in self.fi[j]:
							cij[(i,j)][(ai,aj)] = psuedo_ij - (psuedo_i * self.fi[j][aj])
						else:
							score = psuedo_ij - (psuedo_i ** 2)
							assert(math.isclose(score, 0.0, abs_tol=1e-16))
					else:
						cij[(i,j)][(ai,aj)] = self.fij[(i,j)][(ai,aj)] - (self.fi[i][ai] * self.fi[j][aj])
			
			self.cij = cij
		else:
			raise NotImplementedError
		
		return
	
	
	def cov_matrix(self):
		
		dim = self.length * len(self.q_dca)
		
		l_rescaled = self._rescaled_counts() / self.neff # only want the the rescaled factor
		psuedo_ij = l_rescaled / (len(self.q2) * (1 + l_rescaled))
		psuedo_i  = self.psuedo / (len(self.q) * (1 + self.psuedo))
		
		cov = np.full((dim, dim), psuedo_ij - (psuedo_i ** 2)) 
		
		q_red = len(self.q_dca)
		def idx(ii, aa):
			return (ii * q_red) + aa
		
		for i in range(self.length):
			assert((i,i) not in self.cij)
			for ai, A in enumerate(self.q_dca):
				for j in range(i,self.length):
					for aj, B in zip(list(range(ai,len(self.q_dca))), self.q_dca[ai:]):
						if j == i:
							assert((i, j) not in self.cij)
							if B == A:
								assert(ai == aj)
								if A not in self.fi[i]:
									cov[idx(i,ai), idx(j,aj)] = psuedo_i * (1 - psuedo_i)
								else:
									cov[idx(i,ai), idx(j,aj)] = self.fi[i][A] * (1 - self.fi[i][A])
							else:
								if A not in self.fi[i] and B not in self.fi[i]:
									cov[idx(i,ai), idx(j,aj)] = -1.0 * psuedo_i ** 2
								elif A in self.fi[i] and B not in self.fi[i]:
									cov[idx(i,ai), idx(j,aj)] = -1.0 * self.fi[i][A] * psuedo_i
								elif A not in self.fi[i] and B in self.fi[i]:
									cov[idx(i,ai), idx(j,aj)] = -1.0 * psuedo_i * self.fi[i][B]
								elif A in self.fi[i] and B in self.fi[i]:
									cov[idx(i,ai), idx(j,aj)] = -1.0 * self.fi[i][A] * self.fi[i][B]
								else:
									raise ValueError
								
								cov[idx(j,aj), idx(i,ai)] = cov[idx(i,ai), idx(j,aj)]
						else:
							if (i,j) not in self.cij:
								continue
							
							if (A,B) not in self.cij[(i,j)]:
								continue
							
							cov[idx(i,ai), idx(j,aj)] = self.cij[(i,j)][(A,B)]
							cov[idx(j,aj), idx(i,ai)] = self.cij[(i,j)][(A,B)]
		
		self.cov = cov
		return
	
	
	def infer_couplings(self):
		
		C_inv = np.linalg.pinv(self.cov)
		eijs = -C_inv
		
		self.eijs= eijs
		return
"""


















