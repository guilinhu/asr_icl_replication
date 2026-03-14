### Prerequisites
- Access to a machine with GPU (we used l40)

### Dataset downloading
- The `asr_adaptation.py` file should handle the dataset downloading for you

### Installing dependencies
```
pip install -r requirements.txt
```

### Preprocessing code
Preprocessing code is handled in the `asr_adaptation.py` file.

### Training code
No training command needed since this paper does not require training model.

### Evaluation code
- Running Same-Speaker Adaptation
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

- Running Different-Speaker Adaptation
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

### Running new segment duration for same speaker and different speaker adaptation
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
--min_duration 5.0
```

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
--min_duration 5.0
```

### Analyzing Results

Once the `asr_adaptation.py` generates the `all_results_summary.json` files, run `L2Dataset_Results_analysis.ipynb` on those files for result analysis.

---
