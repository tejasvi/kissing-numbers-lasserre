#!/bin/bash
# cert_run7_v3.sh A_CKPT B_CKPT TGT NAME [REDUND]: cert_run7_v1.sh with roundsol_inc3.jl (hybrid + random incremental columns): certificate point W from two dual
# iterates (bound(A) > TGT > bound(B)), exact rounding, LSC verifier (in the package copy with zonal_skip, inside
# roundsol); log $K5/cert_NAME/cert.log; an existing W.jls is reused; DONE when roundsol has exited (any outcome).
K5=/var/shm_disk/k5; source $K5/env.sh
R=$K5/cert_$4
mkdir -p $R/cache/zonalstore $R/cache/sos $R/cache/samples $R/cache/sampledsos
for m in 4 5 6 7; do ln -sfn $K5/zonal/$m $R/cache/zonalstore/$m; done
cp -n $K5/runF20/cache/samples/* $R/cache/samples/
cd $R
if [ ! -f W.jls ]; then
  echo "=== $(date -u +%FT%TZ) make_target_point $1 $2 $3" >> cert.log
  $JULIA --project=$PROJ -t 32 $K5/work/fleet/make_target_point.jl $1 $2 $3 W.jls 2>&1 | cat >> cert.log
  [ -f W.jls ] || { echo "=== no W.jls" >> cert.log; touch DONE; exit 1; }
fi
echo "=== $(date -u +%FT%TZ) roundsol (redundancy ${5:-14})" >> cert.log
K5_RS_REDUND=${5:-14} K5_RS_KERR=1e-40 K5_RS_KRERR=1e-50 K5_RS_PRIMAL=0 nice -n 5 /usr/bin/time -v \
  $JULIA --project=$PROJ -t 32 $K5/work/fleet/roundsol_inc3.jl 7 -1 1//2 14 20 $3 W.jls 2>&1 | cat >> cert.log
echo "=== $(date -u +%FT%TZ) done" >> cert.log
touch DONE
