#!/bin/bash
# usage: zonal_all.sh "n1,n2,.." dmax maxjobs ; run from the run dir containing cache/
ns=$1; dmax=$2; maxj=$3
for n in ${ns//,/ }; do mkdir -p cache/zonalstore/$n; done
mkdir -p cache/samples cache/sos cache/sampledsos logs
for s in $(seq 0 $dmax); do for b in $(seq 0 $((s/2))); do a=$((s-b))
  if [ $a -ne $b ] || [ $((a%2)) -eq 0 ]; then echo "$a $b"; fi; done; done | sort -k1,1nr -k2,2nr > lambdas.txt
# biggest |lambda| first
awk '{print $1+$2, $0}' lambdas.txt | sort -k1,1nr -k2,2nr | cut -d' ' -f2- > lambdas_sorted.txt
cat lambdas_sorted.txt | xargs -P $maxj -L 1 bash -c 'nice -n 10 $JULIA --project=$PROJ -t 1 $ZONAL '"$ns"' $0 $1 > logs/zonal_$0_$1.log 2>&1'
echo ALLDONE
