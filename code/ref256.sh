#!/bin/bash
# ref256.sh SRC_CKPT NAME [ITERS]: ITERS (default 1) 256-bit iterations of the (14,20) optimization from the 128-bit
# checkpoint SRC_CKPT, in $K5/ref256_NAME (the 128-bit SDP cache is converted up at load); stops itself after the
# first chunk. The refined dual iterates are the 256-bit ingredients of make_target_point.jl.
K5=/var/shm_disk/k5; source $K5/env.sh
S=$K5/ck_n5_d14_20_p256_runF20
R=$K5/ref256_$2
N=${3:-1}
mkdir -p $R/cache/zonalstore $R/cache/sos $R/cache/samples $R/cache/sampledsos
for m in 4 5 6; do ln -sfn $K5/zonal/$m $R/cache/zonalstore/$m; done
cp -n $S/cache/samples/* $R/cache/samples/
for f in $S/cache/sampledsos/*; do ln -sfn $(readlink -f $f) $R/cache/sampledsos/$(basename $f); done
ln -sfn $S/sdp_p128.jls $R/sdp_p128.jls
cp $1 $R/ckpt.jls
cd $R
echo "=== $(date -u +%FT%TZ) refinement of $1 at 256 bits, $N iteration(s)" >> solve.log
K5_PRECS=256 nice -n 10 $JULIA --project=$PROJ -t 192 $K5/work/fleet/solve_ckpt.jl 5 14 20 256 1e-30 $N - >> solve.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  grep -aq "^CHUNK iters=" solve.log && { sleep 5; kill $pid; break; }
  sleep 20
done
wait $pid 2>/dev/null
echo "=== $(date -u +%FT%TZ) refinement done" >> solve.log
