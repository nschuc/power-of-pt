from dataclasses import dataclass
import json
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import argparse
from tokenizers import AddedToken
from transformers import AutoTokenizer


@dataclass
class OvernightInstance:
    utterance: str
    semantic_parse: str
    canonical: str


def preprocess(example, target="meaning"):
    if target == "meaning":
        sp = example["semantic_parse"]
        sp = sp.replace("edu.stanford.nlp.sempre.overnight.SimpleWorld.", "SW.")
        example["semantic_parse"] = sp
    elif target == "canonical":
        example["semantic_parse"] = example["canonical"]

    return example


def load_overnight(
    datadir,
    domain,
    num_train=200,
    do_train=True,
    do_predict=False,
    cache_dir=None,
    seed=47,
):
    path = Path(datadir) / f"{domain}.paraphrases.train.tsv"

    dataset = DatasetDict()

    if do_train:
        # Make dev set for early stopping
        dataset = Dataset.from_csv(
            str(path),
            delimiter="\t",
            names=["utterance", "canonical", "semantic_parse"],
            cache_dir=cache_dir,
        )

        num_dev_instances = int(0.2 * (len(dataset) - num_train))
        dataset = dataset.train_test_split(
            train_size=num_train, test_size=num_dev_instances, seed=seed
        )
        dataset["eval"] = dataset.pop("test")

    if do_predict:
        # Make dev set for early stopping
        path = Path(datadir) / f"{domain}.paraphrases.test.tsv"
        dataset["test"] = Dataset.from_csv(
            str(path),
            delimiter="\t",
            names=["utterance", "canonical", "semantic_parse"],
            cache_dir=cache_dir,
        )

    return dataset


def postprocess(example):
    to_replace = [
        "reverse",
        "domain",
        "concat",
        "listValue",
        "getProperty",
        "singleton",
        "ensureNumericProperty",
        "ensureNumericEntity",
        "countComparative",
        "filter",
        "superlative",
        "countSuperlative",
    ]
    for key in to_replace:
        example = example.replace(
            key, f"edu.stanford.nlp.sempre.overnight.SimpleWorld.{key}"
        )

    num_left_paren = sum(1 for c in example if c == "(")
    num_right_paren = sum(1 for c in example if c == ")")
    diff = num_left_paren - num_right_paren
    if diff > 0:
        print("mismatching brackets!")

    return example


def is_error(d):
    return "BADJAVA" in d or "ERROR" in d or d == "null"


def evaluate(domain, predictions, actual, evaluator_path, postprocess=False):
    # predictions = [ postprocess(p) for p in predictions ]
    if postprocess:
        predictions = [postprocess(p) for p in predictions]

    with tempfile.NamedTemporaryFile("w", suffix=".examples") as tf:
        # tf.writelines(line + "\n" for line in predictions)
        tf.writelines(line + "\n" for line in predictions)
        tf.writelines(line + "\n" for line in predictions)
        tf.flush()

        msg = subprocess.check_output(
            ["evaluator/overnight", domain, tf.name], cwd=evaluator_path
        )

    msg = msg.decode("utf-8")
    denotations = [
        line.split("\t")[1]
        for line in msg.split("\n")
        if line.startswith("targetValue\t")
    ]
    errors = [int(is_error(d)) for d in denotations]
    print(np.mean(errors))

    for err, den in zip(errors, denotations):
        if err:
            print(f"Domain: {domain}; Denotation: {den}")

    predicted_den = denotations[: len(predictions)]
    actual_den = denotations[len(predictions) :]
    matches = [int(p == a) for p, a in zip(predicted_den, actual_den)]
    den_acc = np.mean(matches)
    print(f"Denotation Accuracy: {den_acc}")
    breakpoint()
    return den_acc


def validate_tokenization(dataset, model_name="facebook/bart-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

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

        iv = tokenizer.decode(labels["input_ids"], skip_special_tokens=True)
        iv = iv.replace("!", " !")
        # iv = iv.replace("< ", "<")
        # iv = iv.replace("<= ", "<=")
        iv = iv.replace(".size", " .size")
        if iv != examples["semantic_parse"]:
            print(iv)
            print(examples["semantic_parse"])

        if "domain" in examples:
            outputs["domains"] = examples.get("domain")

        return outputs

    dataset = dataset.map(tokenize_function)

    lens = [len(e["labels"]) for d in dataset.values() for e in d]
    print(f"mean ({np.mean(lens)} max ({np.max(lens)})")


def main():
    parser = argparse.ArgumentParser()
    all_domains = "basketball,blocks,calendar,housing,publications,recipes,restaurants,socialnetwork"
    parser.add_argument(
        "--overnight-path", default="/home/toolkit/data/overnightData", type=str
    )
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--domains", default=all_domains, type=str)

    args = parser.parse_args()
    ds = [
        load_overnight(args.overnight_path, d, do_train=True, do_predict=True)
        for d in args.domains.split(",")
    ]

    predictions = [json.loads(s) for s in args.eval_path.open().readlines()]
    predictions, actual = zip(
        *[(d.get("predicted"), d.get("actual")) for d in predictions]
    )

    for dataset, domain in zip(ds, all_domains.split(",")):
        print(f"domain: {domain}")
        validate_tokenization(dataset)
        predictions = [d.get("semantic_parse") for d in dataset["test"]]
        evaluate(domain, predictions, actual, args.evaluator)


if __name__ == "__main__":
    main()
