## [The Power of Prompt Tuning for Low-resource Semantic Parsing](https://arxiv.org/abs/2110.08525)

Code for reproducing the results of [this paper](https://arxiv.org/abs/2110.08525).


### Install
```bash
git clone https://github.com/nschuc/power-of-pt.git
pip install -r requirements.txt
```

### Training prompts
```bash
python run_prompt_tuning.py --do-train --do-predict \
		--model-name $(MODEL) --num-epochs 5000 --patience 20 \
		--max-gpu-bs 16 --max-eval-gpu-bs 16 \
		--data-dir $(DATA_DIR) --source-domains $(DOMAIN) \
		--dataset $(DATASET) --num-train 200 \
		--split-seed 50 --seed $(SEED) \
		--log-every 50 --eval-every 200 \
		--adafactor --batch-size 32 --lr 0.3 \
		--prompt-length 150 \
		--output-dir ./logs/${JOB_ID}
```

### Constrained Decoding

`prompt_tuning/constrained.py` implements an `allowed_tokens_fn` that can be passed to the HF model generate function as `prefix_allowed_tokens_fn`.