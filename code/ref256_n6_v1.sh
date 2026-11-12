#!/bin/bash
# ref256_n6_v1.sh SRC_CKPT NAME [ITERS] [THREADS]: ref256.sh for n = 6 (copy of ref256_n7_v1.sh): ITERS (default 1) 256-bit iterations of the n = 6
# (14,20) optimization from the checkpoint SRC_CKPT (a kept copy in ck_n6_d14_20_p256_runF20/keep), in $K5/ref256_NAME;
# stops after the first chunk (SIGTERM, then SIGKILL after 2 min: these solves can sit in one long step). The refined
# dual iterates are the inputs of make_target_point.jl.
K5=/var/shm_disk/k5; source $K5/env.sh
S=$K5/ck_n6_d14_20_p256_runF20
R=$K5/ref256_$2
N=${3:-1}; TH=${4:-192}
mkdir -p $R/cache/zonalstore $R/cache/sos $R/cache/samples $R/cache/sampledsos
ln -sfn $K5/zonal6x14 $R/cache/zonalstore/6   # the 490 checked n = 6 files only
cp -n $S/cache/samples/* $R/cache/samples/
for f in $S/cache/sampledsos/*; do ln -sfn $(readlink -f $f) $R/cache/sampledsos/$(basename $f); done
ln -sfn $S/sdp_p128.jls $R/sdp_p128.jls
cp $1 $R/ckpt.jls
cd $R
echo "=== $(date -u +%FT%TZ) refinement of $1 at 256 bits, $N iteration(s), $TH threads" >> solve.log
K5_PRECS=256 nice -n 10 $JULIA --project=$PROJ -t $TH $K5/work/fleet/solve_ckpt.jl 6 14 20 256 1e-30 $N - >> solve.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if grep -aq "^CHUNK iters=" solve.log; then
    sleep 5; kill $pid
    for i in $(seq 1 24); do kill -0 $pid 2>/dev/null || break; sleep 5; done
    kill -0 $pid 2>/dev/null && kill -9 $pid
    break
  fi
  sleep 20
done
wait $pid 2>/dev/null
echo "=== $(date -u +%FT%TZ) refinement done" >> solve.log
