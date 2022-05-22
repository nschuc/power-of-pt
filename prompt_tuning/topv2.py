from pathlib import Path
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric
from transformers.models.auto.tokenization_auto import AutoTokenizer
from pyparsing import OneOrMore, nestedExpr, NoMatch

topGrammar = OneOrMore(nestedExpr("[", "]", ignoreExpr=NoMatch()))


ONTOLOGY_LABELS = [
    "IN:UPDATE_DIRECTIONS",
    "SL:CATEGORY_LOCATION",
    "IN:LOOP_MUSIC",
    "IN:ADD_TO_PLAYLIST_MUSIC",
    "IN:GET_LOCATION_WORK",
    "IN:CANCEL_MESSAGE",
    "IN:GET_LOCATION_HOMETOWN",
    "SL:WAYPOINT_AVOID",
    "SL:MUSIC_PLAYLIST_TITLE",
    "IN:GET_DIRECTIONS",
    "SL:TYPE_CONTENT",
    "SL:OBSTRUCTION_AVOID",
    "SL:PATH_AVOID",
    "SL:LOCATION_CURRENT",
    "SL:CONTACT",
    "SL:PERIOD",
    "IN:PAUSE_TIMER",
    "SL:WEATHER_ATTRIBUTE",
    "SL:DATE_TIME_DEPARTURE",
    "SL:TYPE_RELATION",
    "SL:GROUP",
    "IN:GET_BIRTHDAY",
    "SL:TYPE_REACTION",
    "IN:GET_INFO_ROUTE",
    "SL:MEASUREMENT_UNIT",
    "IN:REMOVE_FROM_PLAYLIST_MUSIC",
    "SL:MUSIC_ARTIST_NAME",
    "IN:LIKE_MUSIC",
    "IN:GET_EVENT_ATTENDEE",
    "SL:SEARCH_RADIUS",
    "SL:TODO",
    "SL:PERSON_REMINDED_REMOVED",
    "SL:TYPE_CONTACT",
    "IN:GET_INFO_TRAFFIC",
    "SL:TYPE_INFO",
    "IN:GET_CONTACT",
    "IN:CREATE_REMINDER",
    "IN:DELETE_REMINDER",
    "IN:SNOOZE_ALARM",
    "IN:REPLAY_MUSIC",
    "IN:GET_REMINDER_AMOUNT",
    "SL:RECURRING_DATE_TIME",
    "IN:PLAY_MUSIC",
    "SL:MUSIC_TRACK_TITLE",
    "SL:METHOD_TIMER",
    "IN:SILENCE_ALARM",
    "SL:MUTUAL_SCHOOL",
    "IN:CREATE_TIMER",
    "IN:IGNORE_MESSAGE",
    "SL:ALARM_NAME",
    "IN:GET_INFO_ROAD_CONDITION",
    "SL:PERSON_REMINDED",
    "IN:GET_ESTIMATED_DEPARTURE",
    "IN:GET_DISTANCE",
    "IN:GET_EVENT_ORGANIZER",
    "SL:MUSIC_RADIO_ID",
    "SL:LOCATION_HOME",
    "IN:DISLIKE_MUSIC",
    "IN:NEGATION",
    "SL:CONTACT_RELATED",
    "SL:ROAD_CONDITION",
    "IN:GET_SUNSET",
    "SL:DATE_TIME",
    "IN:START_SHUFFLE_MUSIC",
    "IN:GET_LOCATION_HOME",
    "SL:POINT_ON_MAP",
    "SL:DATE_TIME_ARRIVAL",
    "SL:ORGANIZER_EVENT",
    "SL:ATTENDEE_ADDED",
    "SL:AGE",
    "SL:UNIT_DISTANCE",
    "IN:PREVIOUS_TRACK_MUSIC",
    "SL:DATE_TIME_RECURRING",
    "SL:DESTINATION",
    "SL:LOCATION",
    "IN:SELECT_ITEM",
    "IN:UPDATE_TIMER",
    "IN:UNSUPPORTED_MESSAGING",
    "SL:LOCATION_MODIFIER",
    "IN:SKIP_TRACK_MUSIC",
    "IN:GET_INFO_CONTACT",
    "IN:GET_ESTIMATED_ARRIVAL",
    "SL:PERSON_REMINDED_ADDED",
    "IN:DELETE_ALARM",
    "IN:UPDATE_ALARM",
    "IN:SEND_TEXT_MESSAGE",
    "SL:RESOURCE",
    "SL:CATEGORY_EVENT",
    "IN:REPLY_MESSAGE",
    "IN:GET_EVENT_ATTENDEE_AMOUNT",
    "SL:MUTUAL_EMPLOYER",
    "SL:MUTUAL_LOCATION",
    "IN:GET_REMINDER_DATE_TIME",
    "SL:NAME_APP",
    "IN:RESUME_TIMER",
    "SL:RECIPIENT",
    "IN:HELP_REMINDER",
    "IN:UNSUPPORTED_EVENT",
    "SL:WAYPOINT",
    "SL:BIRTHDAY",
    "SL:CONTENT_EXACT",
    "IN:UPDATE_REMINDER_DATE_TIME",
    "IN:GET_WEATHER",
    "IN:GET_TIME",
    "IN:GET_TODO",
    "SL:METHOD_RETRIEVAL_REMINDER",
    "IN:UNSUPPORTED_NAVIGATION",
    "SL:ATTENDEE_REMOVED",
    "SL:NAME_EVENT",
    "IN:GET_LOCATION_SCHOOL",
    "IN:SEND_MESSAGE",
    "SL:SENDER",
    "SL:TODO_NEW",
    "IN:DELETE_TIMER",
    "SL:DATE_TIME_NEW",
    "SL:WEATHER_TEMPERATURE_UNIT",
    "IN:GET_MESSAGE",
    "IN:CREATE_ALARM",
    "IN:UNSUPPORTED_WEATHER",
    "IN:GET_ESTIMATED_DURATION",
    "SL:DATE_TIME_BIRTHDAY",
    "SL:DURATION",
    "SL:ATTENDEE",
    "IN:GET_LOCATION",
    "IN:CREATE_PLAYLIST_MUSIC",
    "IN:STOP_MUSIC",
    "SL:TAG_MESSAGE",
    "IN:ADD_TIME_TIMER",
    "IN:GET_RECURRING_DATE_TIME",
    "SL:RECURRING_DATE_TIME_NEW",
    "IN:GET_SUNRISE",
    "SL:CONTENT_EMOJI",
    "SL:AMOUNT",
    "IN:GET_REMINDER",
    "IN:UPDATE_REMINDER",
    "IN:REACT_MESSAGE",
    "SL:JOB",
    "SL:SOURCE",
    "IN:RESTART_TIMER",
    "SL:ORDINAL",
    "SL:METHOD_TRAVEL",
    "SL:FREQUENCY",
    "IN:GET_ALARM",
    "SL:MUSIC_TYPE",
    "SL:TIMER_NAME",
    "IN:UNSUPPORTED_MUSIC",
    "IN:GET_TIMER",
    "SL:PATH",
    "SL:WAYPOINT_ADDED",
    "IN:GET_REMINDER_LOCATION",
    "SL:MUSIC_GENRE",
    "SL:LOCATION_USER",
    "IN:UPDATE_REMINDER_TODO",
    "IN:GET_EVENT",
    "SL:ROAD_CONDITION_AVOID",
    "IN:SUBTRACT_TIME_TIMER",
    "SL:TIME_ZONE",
    "SL:MUSIC_ALBUM_TITLE",
    "SL:ATTRIBUTE_EVENT",
    "IN:UNSUPPORTED_TIMER",
    "SL:MUSIC_PROVIDER_NAME",
    "IN:UNSUPPORTED_ALARM",
    "SL:ATTENDEE_EVENT",
    "SL:LOCATION_WORK",
    "IN:SET_DEFAULT_PROVIDER_MUSIC",
    "IN:PAUSE_MUSIC",
]


def lf_to_str(lf):
    if isinstance(lf, str):
        return lf
    return "[" + " ".join(lf_to_str(t) for t in lf) + " ]"


def canonicalize(tree, add_token=None):
    label, *tokens = tree
    if add_token is not None:
        add_token([label])

    children = [
        canonicalize(child, add_token=add_token)
        for child in tokens
        if isinstance(child, list)
    ]

    if not children:
        return tree

    return [label, *children]


def preprocess(example, add_token=None):
    lf = example["semantic_parse"]
    try:
        tree = topGrammar.parseString(lf).asList()[0]
    except Exception as e:
        print("pyparsing error")
        print(example)
        print(e)
        return example

    converted = canonicalize(tree, add_token=add_token)
    lf = lf_to_str(converted)

    example["semantic_parse"] = lf
    return example


def shorten_ontology_labels(example):
    lf = example["semantic_parse"]

    for idx, label in enumerate(ONTOLOGY_LABELS):
        lf = lf.replace(label, f"T{idx}")

    example["semantic_parse"] = lf
    return example


def load_top(
    datadir,
    sources,
    target_spis=None,
    do_train=False,
    do_predict=False,
    cache_dir="/transformers_cache",
):
    datadir = Path(datadir)

    dataset = DatasetDict()

    if do_train:
        spi_name = ""

        traindir = datadir
        if target_spis:
            traindir = datadir / "low_resource_splits"
            spi_name = f"_{target_spis}spis"

        filenames = [
            str(traindir / f"{domain}_train{spi_name}.tsv") for domain in sources
        ]
        dataset["train"] = Dataset.from_csv(
            filenames, delimiter="\t", cache_dir=cache_dir
        )

        filenames = [
            str(traindir / f"{domain}_eval{spi_name}.tsv") for domain in sources
        ]
        dataset["eval"] = Dataset.from_csv(
            filenames, delimiter="\t", cache_dir=cache_dir
        )

        if target_spis:
            dataset = dataset.rename_column("seqlogical", "semantic_parse")

    if do_predict:
        filenames = [str(datadir / f"{domain}_test.tsv") for domain in sources]
        dataset["test"] = Dataset.from_csv(
            filenames, delimiter="\t", cache_dir=cache_dir
        )

    return dataset


if __name__ == "__main__":
    sources = [
        "alarm",
        "messaging",
        "music",
        "timer",
        "navigation",
        "event",
        "reminder",
        "weather",
    ]

    dataset = load_top(
        "/home/toolkit/data/TOPv2_Dataset",
        sources=sources,
        do_train=True,
        do_predict=True,
        cache_dir="/tmp",
    )
    tokenizer = AutoTokenizer.from_pretrained("t5-large")

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

        if "domain" in examples:
            outputs["domains"] = examples.get("domain")

        return outputs

    ontology_tokens = set()
    add_token = lambda tokens: ontology_tokens.update(tokens)
    dataset.map(lambda example: preprocess(example, add_token=add_token))