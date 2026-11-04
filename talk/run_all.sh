#!/usr/bin/env bash
# Run all experiment batches. Convergence/iteration batches in parallel (timings
# there are not used for performance claims), then the timing and work-precision
# batches alone.
set -u
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
LOG=results/logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "start $(date)" > "$LOG/run_all.status"
$PY -u experiments.py --batch h > "$LOG/h.log" 2>&1 &
$PY -u experiments.py --batch eps > "$LOG/eps.log" 2>&1 &
( $PY -u experiments.py --batch iter_eps > "$LOG/iter_eps.log" 2>&1
  $PY -u experiments.py --batch iter_h > "$LOG/iter_h.log" 2>&1 ) &
( $PY -u experiments.py --batch cont_cmp > "$LOG/cont_cmp.log" 2>&1
  $PY -u experiments.py --batch balance > "$LOG/balance.log" 2>&1 ) &
wait
echo "parallel batches done $(date)" >> "$LOG/run_all.status"

$PY -u experiments.py --batch timing > "$LOG/timing.log" 2>&1
$PY -u experiments.py --batch workprec > "$LOG/workprec.log" 2>&1
echo "all done $(date)" >> "$LOG/run_all.status"
