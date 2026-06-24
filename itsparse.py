#!/usr/bin/python3

import argparse
from itertools import product
import json
import matplotlib.pyplot as plt
import sys

import cppyy
import numpy as np

parser = argparse.ArgumentParser(description="measure mutual information with new psuedo counts")

parser.add_argument('--aln', '-a', required=True, type=str, metavar='<str>', help="path to alignment")
parser.add_argument('--pdb', '-p', required=True, type=str, metavar='<str>', help="path to pdb structure")
parser.add_argument('--sim', '-s', default=0.8, type=float, metavar='<float>', help="max sequence identity for sequence clustering")
parser.add_argument('--pseudo', '-l', default=1.0, type=float, metavar='<float>', help="pseudo count scale. psuedo counts added as l * n_eff")

args = parser.parse_args()

q = [
	'A','C','D','E','F',
	'G','H','I','K','L',
	'M','N','P','Q','R',
	'S','T','V','W','Y',
	'-'
]
qmap = {aa:i for i, aa in enumerate(q)}
q2 = list(product(q, q))

# alignment
seqs = []
with open(args.aln, 'r') as fp:
	seqs = [line.rstrip() for line in fp]

#seqs = seqs[:1000]

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

depth = len(seqs)
length = len(seqs[0])

results = np.zeros(depth, dtype=np.intc)
lens = np.array([length] * depth, dtype=np.intc)

cppyy.gbl.get_ma(seqs, lens, depth, args.sim, results)

ma = dict()
ma = {k:int(v) for k, v in enumerate(results)}

neff = 0.0
for k,v in ma.items(): neff += 1.0/v

print(f"depth: {depth} neff: {neff:.2f}")

pcs = args.pseudo * neff
fi_pf = pcs / (len(q) * (pcs + neff))

print(f"pcs: {pcs:.2f} fi_pf: {fi_pf:1.4E}")

fi = np.full((length, len(q)), fi_pf)

incf = 1.0 / (pcs + neff)
for i in range(length):
	for j in range(depth):
		aa = seqs[j][i]
		if aa not in q: continue
		fi[i, qmap[aa]] += incf * (1/ma[j])

pij = (args.pseudo ** 2) / (1 + (2 * args.pseudo))
pcs_ij = pij * neff
fij_pf = pcs_ij / (len(q2) * (pcs_ij + neff))

fij = np.full((length, length, len(q), len(q)), fij_pf)

incf = 1.0 / (pcs + neff)
for i in range(length):
	for j in range(i,length):
		for k in range(depth):
			a = seqs[k][i]
			b = seqs[k][j]
			
			if a not in q or b not in q: continue
			
			fij[i, j, qmap[a], qmap[b]] += incf * (1/ma[k])
			if i != j:
				fij[j, i, qmap[b], qmap[a]] += incf * (1/ma[k])

qlen = len(q)
cov = np.zeros((qlen * length, qlen * length))
for i in range(length):
	for j in range(i,length):
		for ai in range(len(q)):
			for aj in range(len(q)):
				idi = i*qlen + ai
				idj = j*qlen + aj
				
				cov[idi, idj] = fij[i,j,ai,aj] - (fi[i,ai] * fi[j,aj])
				cov[idj, idi] = fij[j,i,aj,ai] - (fi[j,aj] * fi[i,ai])


idx = np.arange(qlen * length, dtype=int)
res_i = idx // len(q)

dist = np.abs(res_i[:, None] - res_i[None, :])
band_mask = (dist < 6) & (dist > 0)

mask = np.where(np.isclose(cov, fij_pf - fi_pf**2))
print(mask)

plt.imshow(cov, vmin=-np.max(np.abs(cov)), vmax=np.max(np.abs(cov))cmap="viridis")
plt.show()

cov += 1e-6 * np.eye(cov.shape[0])
K = np.linalg.inv(cov)

plt.imshow(K, cmap="viridis")
plt.show()

K[band_mask] *= 0.9
K[mask] *= 0.1

plt.imshow(K, cmap="viridis")
plt.show()

K += 1e-6 * np.eye(K.shape[0])
cov1 = np.linalg.inv(K)

plt.imshow(cov1, cmap="viridis")
plt.show()

diff = np.abs(cov - cov1)
plt.imshow(diff, cmap="viridis")
plt.show()

alpha = 0.1
cov1 = (1-alpha)*cov + alpha*cov1

K1 = np.linalg.inv(cov1)
plt.imshow(K1, cmap="viridis")
plt.show()

K1[band_mask] *= 0.1
K1[mask] = 1e-6

plt.imshow(K1, cmap="viridis")
plt.show()

diffK = np.abs(K1 - K)
plt.imshow(diffK, cmap="viridis")
plt.show()

sys.exit()






















