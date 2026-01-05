#!/bin/bash
set -euo pipefail

ARRAY_JOBID_RAW=$(qsub llama_eval_array.pbs)
echo "Array job id: $ARRAY_JOBID_RAW"

MERGE_JOBID=$(qsub -W depend=afterok:${ARRAY_JOBID_RAW} llama_merge_plot.pbs)
echo "Merge job id: $MERGE_JOBID"
