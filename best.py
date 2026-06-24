#!/usr/bin/python3

import json
import sys

from msalib import MSA
from msalib import read_stockholm

neffs = [("",0.0)] * 10

for msa in read_stockholm(sys.argv[1]):
	#print(msa.accession)
	
	msa._seq_cluster(sim=0.80)
	msa.set_neff()
	neff = float(msa.neff)
	
	neffs = sorted(neffs, key=lambda x: x[1], reverse=True)
	#print(neffs)
	#print(neffs[9])
	
	if neff > neffs[9][1]:
		neffs[9] = (msa.identifier, neff)
	
	#print(json.dumps(neffs,indent=2))
	#sys.exit()
	

print("Top 10 N_eff @ 80% Sequence Identity among Pfam families")
for i, (identifier, neff) in enumerate(sorted(neffs, key=lambda x: x[1], reverse=True)):
	print(f"{i+1}\t {identifier:<15} {neff:6.2f}")