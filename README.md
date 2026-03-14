### Prerequisites
- Access to a machine with GPU (we used A100s)

### Dataset downloading
- The `asr_adaptation.py` file should handle the dataset downloading for you

### Installing dependencies
```
pip install -r requirements.txt
```

### Running Same-Speaker Adaptation
- for same speaker few shot evaluation, run
```
python asr_adaptation.py \
--output_dir /nlp/scr/$USER/20250513_shot_results \
--cache_dir /nlp/scr/$USER/.cache/huggingface \
--max_shots 12 \
--max_trials 50 \
--dataset all \
--speaker_condition same \
--prompt_type both \
--seed 42 \
```

### Running Different-Speaker Adaptation

```
python asr_adaptation.py \
--output_dir /nlp/scr/$USER/20250513_shot_results \
--cache_dir /nlp/scr/$USER/.cache/huggingface \
--max_shots 12 \
--max_trials 50 \
--dataset all \
--speaker_condition different \
--prompt_type both \
--seed 42 \
```

### Analyzing Results

Once the `asr_adaptation.py` generates the `all_results_summary.json` files, run `ASR_Adaptation_analysis.ipynb` on those files for result analysis.

---
