
## Join + verify after cloning

The checkpoint files were split into parts parts dues to upload limit. Use below command to join the files back.

```bash
python3 join_parts.py \
  --parts_dir RLGOAL/Evaluation/len8000plus_ep62944_env0_len16377_ep62944.pt.parts \
  --out_file  RLGOAL/Evaluation/len8000plus_ep62944_env0_len16377_ep62944.pt
```
