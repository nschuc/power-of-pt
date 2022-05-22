from tokenizers import AddedToken
from transformers import T5TokenizerFast
from transformers.models.auto.configuration_auto import AutoConfig
import argparse
import inspect
import itertools
import os
import datasets
import wandb
import json

import torch
import tqdm
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.optimization import Adafactor, AdamW
import numpy as np

from accelerate import Accelerator
import prompt_tuning.overnight as overnight
from prompt_tuning import topv2
from prompt_tuning.constrained import build_prefix_allowed_tokens_fn
from prompt_tuning.wrapper import PromptWrapper
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

SOURCE_DOMAINS = [
    "alarm",
    "messaging",
    "music",
    "timer",
    "navigation",
    "event",
    "reminder",
    "weather",
]

CACHE_DIR = "/transformers_cache"
MAX_GPU_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64


def create_optimizer(args, params):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}

    optimizer_kwargs["lr"] = args.lr
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def load_dataset(args, tokenizer):
    def tokenize_function(examples):
        outputs = tokenizer(examples["utterance"], truncation=True, max_length=None)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["semantic_parse"],
                max_length=None,
                padding=False,
                truncation=True,
                return_overflowing_tokens=False,
            )

        outputs["labels"] = labels["input_ids"]

        return outputs

    source_domains = args.source_domains.split(",")

    if args.dataset == "topv2":
        dataset = topv2.load_top(
            datadir=args.data_dir,
            sources=source_domains,
            target_spis=args.target_spis,
            do_train=args.do_train,
            do_predict=args.do_predict,
        )

        if args.canon_utter:
            dataset = dataset.map(topv2.preprocess)

        if args.canon_token:
            dataset = dataset.map(topv2.shorten_ontology_labels)

        dataset = dataset.map(
            tokenize_function,
            batched=True,
        )

    elif args.dataset == "overnight":
        domain = source_domains[0]
        dataset = overnight.load_overnight(
            datadir=args.data_dir,
            domain=domain,
            do_train=args.do_train,
            do_predict=args.do_predict,
            num_train=args.num_train,
            seed=args.split_seed or args.seed,
        )

        target = "canonical" if args.overnight_canonical else "meaning"
        dataset = dataset.map(lambda e: overnight.preprocess(e, target=target))

        dataset = dataset.map(
            tokenize_function,
            batched=True,
        )

    return dataset


def resume_from_checkpoint(checkpoint, model):
    print(f"resuming from {checkpoint}")

    if checkpoint is None:
        return 1, -100, 1

    try:
        state_dict = torch.load(checkpoint)
    except FileNotFoundError:
        return 1, -100, 1

    model.load_state_dict(state_dict["model"], strict=False)

    return state_dict["epoch"] + 1, state_dict["eval_metric"], state_dict.get("step", 1)


def save_checkpoint(path, model, optimizer, eval_metric, epoch, step, keep):
    if path is None:
        return

    state = {
        "model": {k: v for k, v in model.state_dict().items() if keep(k)},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "eval_metric": eval_metric,
    }
    torch.save(state, path)


def remove_unused_columns(model, dataset: "datasets.Dataset", wrapper=None):
    # Inspect model forward signature to keep only the arguments it accepts.
    wrapper_params = (
        list(inspect.signature(wrapper.forward).parameters.keys()) if wrapper else []
    )
    signature = inspect.signature(model.forward)
    _signature_columns = (
        list(signature.parameters.keys()) + ["label", "label_ids"] + wrapper_params
    )

    db_columns = dataset.column_names
    if isinstance(db_columns, dict):
        db_columns = itertools.chain(*db_columns.values())

    ignored_columns = list(set(db_columns) - set(_signature_columns))
    print(f"removing columns {ignored_columns}")
    return dataset.remove_columns(ignored_columns)


def do_train(args):
    # Initialize accelerator
    accelerator = Accelerator(device_placement=False, fp16=args.fp16, cpu=args.cpu)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)

    tokenizer_name_or_path = (
        "/experiments/prompt-tuning/ontology-tokenizer"
        if args.canon_vocab
        else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    dataset = load_dataset(args, tokenizer)

    train_dataset = dataset.get("train")
    eval_dataset = dataset.get("eval")
    test_dataset = dataset.get("test")

    if args.debug:
        train_dataset = train_dataset.select(range(64))
        eval_dataset = eval_dataset.select(range(64))

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    max_gpu_batch_size = args.max_gpu_bs or MAX_GPU_BATCH_SIZE
    if batch_size > max_gpu_batch_size:
        gradient_accumulation_steps = batch_size // max_gpu_batch_size
        batch_size = max_gpu_batch_size

    set_seed(args.seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    if args.init_random:
        config = AutoConfig.from_pretrained(
            args.model_name,
            return_dict=True,
        )
        lm = AutoModelForSeq2SeqLM.from_config(config)
    else:
        lm = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            return_dict=True,
        )

    lm.resize_token_embeddings(len(tokenizer))

    model = PromptWrapper(
        lm,
        prompt_length=args.prompt_length,
        domains=SOURCE_DOMAINS,
        initialize_from_vocab=args.init_from_vocab,
    )

    train_dataset, eval_dataset, test_dataset = [
        remove_unused_columns(lm, d, wrapper=model) if d else None
        for d in (train_dataset, eval_dataset, test_dataset)
    ]

    # Constrained decoding as in Shin. et al
    prefix_allowed_tokens_fn = None
    valid_targets = datasets.concatenate_datasets([*dataset.values()])["labels"]
    max_seq_len = max(map(len, valid_targets))
    if args.constrained:
        prefix_allowed_tokens_fn, is_in_trie = build_prefix_allowed_tokens_fn(
            valid_targets, tokenizer.eos_token_id
        )

    model = model.to(accelerator.device)

    def keep(key):
        if key.startswith("model"):
            return args.train_lm
        if key.startswith("prompt"):
            return True

    trainable_params = [(key, p) for key, p in model.named_parameters() if keep(key)]

    optimizer = create_optimizer(args, trainable_params)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=(
            -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        ),
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    def collate_fn(features):
        features = data_collator(features)
        return features

    # Instantiate dataloaders.
    train_dataloader = (
        DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
        )
        if args.do_train
        else None
    )
    eval_dataloader = (
        DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.max_eval_gpu_bs,
        )
        if args.do_train
        else None
    )
    test_dataloader = (
        DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.max_eval_gpu_bs,
        )
        if args.do_predict
        else None
    )

    # Prepare everything
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    start_epoch, best_eval_metric, step = resume_from_checkpoint(
        args.resume_from or f"{args.output_dir}/best.pth", model
    )
    best_eval_metric = best_eval_metric if not args.resume_from else -100

    best_val_metric, best_val_metric_step = 0, 0
    global_step = 0

    # if the run was interrupted we want to continue from checkpoint epoch otherwise this is fine-tuning
    start_epoch = start_epoch if not args.resume_from else 1

    # Now we train the model
    if args.do_train:
        for epoch in range(start_epoch, num_epochs):
            steps_since_best_val_metric = global_step - best_val_metric_step

            if steps_since_best_val_metric > args.patience * args.eval_every:
                accelerator.print(
                    f"Early stopping! {steps_since_best_val_metric} > {args.patience * args.eval_every} steps without improvement"
                )
                break

            progress_bar = tqdm(
                train_dataloader,
                disable=not accelerator.is_local_main_process,
                desc="training",
            )
            for step, batch in enumerate(progress_bar):
                model.train()
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()

                if global_step % args.log_every == 0:
                    model.eval()
                    with torch.no_grad():
                        max_length = batch["labels"].shape[-1]
                        generated_tokens = model.generate(
                            batch["input_ids"], max_length=max_length
                        )
                        generated_tokens = accelerator.gather(generated_tokens)

                    predicted_parse = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    label_ids = accelerator.gather(batch["labels"])
                    _label_ids = torch.where(
                        label_ids != -100, label_ids, tokenizer.pad_token_id
                    )
                    labels = tokenizer.batch_decode(
                        _label_ids, skip_special_tokens=True
                    )
                    em_accuracy = np.mean(
                        [
                            pred == actual
                            for pred, actual in zip(predicted_parse, labels)
                        ]
                    )
                    inputs = tokenizer.batch_decode(
                        batch["input_ids"], skip_special_tokens=True
                    )

                    for pred, actual, input in zip(predicted_parse, labels, inputs):
                        accelerator.print("\n\n")
                        accelerator.print(f"Input:\n{input}")
                        accelerator.print(f"Predicted:\n{pred}")
                        accelerator.print(f"Actual:\n{actual}")

                    if accelerator.is_local_main_process:
                        wandb.log(
                            {"train/loss": loss, "train/exact_match": em_accuracy}
                        )
                        accelerator.print(f"loss step {global_step}: {loss}")
                        accelerator.print(
                            f"em accuracy step {global_step}: {em_accuracy}"
                        )

                if global_step % args.eval_every == 0:
                    model.eval()
                    accuracy = []
                    losses = []
                    predictions = []
                    eval_bar = tqdm(
                        eval_dataloader,
                        disable=not accelerator.is_local_main_process,
                        desc="evaluation",
                    )
                    for step, batch in enumerate(eval_bar):
                        with torch.no_grad():
                            batch.to(accelerator.device)
                            max_length = batch["labels"].shape[-1]
                            generated_tokens = model.generate(
                                **batch, max_length=max_length
                            )
                            generated_tokens = accelerator.gather(generated_tokens)

                            outputs = model(**batch)
                            outputs = accelerator.gather(outputs)

                            loss = (
                                (
                                    outputs["loss"]
                                    if isinstance(outputs, dict)
                                    else outputs[0]
                                )
                                .mean()
                                .detach()
                                .cpu()
                            )

                        predicted_parse = tokenizer.batch_decode(
                            generated_tokens, skip_special_tokens=True
                        )

                        label_ids = accelerator.gather(batch["labels"])
                        _label_ids = torch.where(
                            label_ids != -100, label_ids, tokenizer.pad_token_id
                        )
                        labels = tokenizer.batch_decode(
                            _label_ids, skip_special_tokens=True
                        )
                        accuracy.extend(
                            (
                                pred == actual
                                for pred, actual in zip(predicted_parse, labels)
                            )
                        )
                        predictions.extend(
                            dict(predicted=p, actual=a)
                            for p, a, in zip(predicted_parse, labels)
                        )
                        losses.append(loss)

                    eval_metric = np.mean(accuracy)
                    eval_loss = np.mean(losses)

                    if eval_metric > best_val_metric:
                        best_val_metric = eval_metric
                        best_val_metric_step = global_step

                    # Use accelerator.print to print only on the main process.
                    accelerator.print(
                        f"epoch {epoch}: ({eval_metric} acc) ({eval_loss} loss)",
                        eval_metric,
                    )
                    if accelerator.is_local_main_process:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "eval/exact_match": eval_metric,
                                "eval/loss": eval_loss,
                            }
                        )

                    with open(
                        f"{args.output_dir}/predictions_eval_{epoch}.jsonl", "w"
                    ) as f:
                        f.writelines(json.dumps(d) + "\n" for d in predictions)

                    accelerator.print(json.dumps(predictions[0]))

                    path = f"{args.output_dir}/epoch-{global_step}.pth"
                    if args.save_all:
                        save_checkpoint(
                            path, model, optimizer, eval_metric, epoch, step, keep
                        )

                    if eval_metric > best_eval_metric:
                        best_eval_metric = eval_metric
                        path = f"{args.output_dir}/best.pth"
                        save_checkpoint(
                            path, model, optimizer, eval_metric, epoch, step, keep
                        )
                    model.train()

    _, _, _ = resume_from_checkpoint(f"{args.output_dir}/best.pth", model)

    if args.do_predict:
        accelerator.print("Evaluating on test set")
        model.eval()
        all_predictions = []
        accuracy = []
        progress_bar = tqdm(
            test_dataloader, disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(progress_bar):
            with torch.no_grad():
                batch.to(accelerator.device)
                max_length = batch["labels"].shape[-1]

                output = model.generate(
                    **batch,
                    max_length=max_seq_len + 1,
                    num_beams=args.num_beams,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    return_dict_in_generate=True,
                )
                sequences = accelerator.gather(output.sequences)

            if args.constrained:
                assert all(is_in_trie(s[1:].tolist()) for s in sequences)

            generated_tokens = sequences

            predicted_parse = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            label_ids = accelerator.gather(batch["labels"])
            _label_ids = torch.where(
                label_ids != -100, label_ids, tokenizer.pad_token_id
            )
            labels = tokenizer.batch_decode(_label_ids, skip_special_tokens=True)

            accuracy.extend(
                (pred == actual for pred, actual in zip(predicted_parse, labels))
            )
            all_predictions.extend(
                dict(predicted=p, actual=a) for p, a, in zip(predicted_parse, labels)
            )

        accuracy = np.mean(accuracy)
        accelerator.print(f"Test accuracy: {accuracy}")

        constrained_key = "_constrained" if args.constrained else ""
        decoding_spec = constrained_key + f"_{args.num_beams}-beams"
        wb_metric_key = f"test/exact_match" + decoding_spec
        if wandb.run is None and args.uid:
            api = wandb.Api()
            run = api.run(f"nschuc/prompt-tuning/{args.uid}")
            run.summary[wb_metric_key] = accuracy
            run.summary.update()
        else:
            wandb.run.summary[wb_metric_key] = accuracy

        with open(f"{args.output_dir}/test-predictions{decoding_spec}.jsonl", "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in all_predictions)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--model-name", type=str, default="t5-small", help="Name of model to load"
    )
    parser.add_argument("--uid", type=str, help="Job id for logging test metric")
    parser.add_argument(
        "--dataset", type=str, default="topv2", help="Name of dataset to load"
    )
    parser.add_argument("--output-dir", type=str, help="Directory to store artifacts")
    parser.add_argument("--data-dir", type=str, help="Directory of the data files")
    parser.add_argument(
        "--resume-from", type=str, help="Checkpoint path to continue from"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.3, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split-seed", type=int, default=None, help="Seed for generating data split"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Seed for generating data split"
    )
    parser.add_argument("--max-gpu-bs", type=int, default=32, help="Random seed")
    parser.add_argument("--max-eval-gpu-bs", type=int, default=64, help="Random seed")
    parser.add_argument(
        "--num-epochs", type=int, default=20, help="Number of epochs to train for"
    )
    parser.add_argument("--num-shards", type=int, help="Number of dataset shards")
    parser.add_argument(
        "--shard-idx", type=int, help="index of dataset shard to train on"
    )
    parser.add_argument(
        "--target-spis",
        type=int,
        help="SPI target split to use (samples per intent slot)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=200,
        help="Number of training examples to load (overnight)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=0,
        help="Number of prefix embeddings to learn",
    )
    parser.add_argument(
        "--source-domains",
        type=str,
        default=",".join(SOURCE_DOMAINS),
        help="Domains to use for source-training",
    )
    parser.add_argument(
        "--log-every", type=int, default=50, help="Frequency with which to log"
    )
    parser.add_argument(
        "--eval-every", type=int, default=200, help="Frequency with which to evaluate"
    )
    parser.add_argument(
        "--ignore-pad-token-for-loss",
        type=bool,
        default=True,
        help="Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="If passed runs two batch epochs"
    )
    parser.add_argument("--do-train", action="store_true", help="Run training loop")
    parser.add_argument(
        "--freeze-prompt",
        action="store_true",
        help="Freeze prompt weights for fine-tuning",
    )
    parser.add_argument(
        "--train-lm", action="store_true", help="Optimize LM weights as well"
    )
    parser.add_argument("--do-predict", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--fp16", type=bool, default=False, help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--init-random", action="store_true", help="Don't load pre-trained weights"
    )
    parser.add_argument(
        "--init-from-vocab",
        action="store_true",
        help="Initialize prompts from pretrained vocab",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="If passed, will save a checkpoint at every epoch",
    )
    parser.add_argument(
        "--canon-utter",
        action="store_true",
        help="Remove extraneous utterance tokens from logical form (TOPv2)",
    )
    parser.add_argument(
        "--canon-token", action="store_true", help="Shorten ontology tokens"
    )
    parser.add_argument(
        "--canon-vocab",
        action="store_true",
        help="Whether to add domain tokens to vocab",
    )
    parser.add_argument(
        "--overnight-canonical",
        action="store_true",
        help="Whether to target overnight canonical form or meaning representation",
    )
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Constrain decoding to targets in the dataset",
    )
    parser.add_argument(
        "--num-beams", type=int, default=1, help="Number of beams to use when decoding"
    )
    parser.add_argument(
        "--adafactor",
        action="store_true",
        help="Switch optimizer to AdaFactor instead of AdamW",
    )
    parser.add_argument(
        "--cpu", type=bool, default=False, help="If passed, will train on the CPU."
    )
    args = parser.parse_args()

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        run = wandb.init(project="power-repro", resume="allow", id=args.uid)
        wandb.config.update(args, allow_val_change=True)
        run.define_metric("eval/exact_match", summary="max")

    if args.do_train or args.do_predict:
        do_train(args)


if __name__ == "__main__":
    main()
