from pathlib import Path


ca_housing = "california_house_prices"
jc_penney = "jc_penney_products"
online_boats = "online_boat_listings"
online_boats_no_foreign_languages = "online_boat_listings_no_foreign_languages"
test_dataset = "test"

SEED = 97
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
TOKENISER_CONFIG = "tokeniser"
FEATURISER_CONFIG = "featuriser"
MODEL_CONFIG = "model"
PREPROCESSER_ORDER = "preprocesser_order"
FINETUNE_OPT = "finetune"
MANDATORY_CONFIG_KEYS = [
    TOKENISER_CONFIG,
    FEATURISER_CONFIG,
    MODEL_CONFIG,
    PREPROCESSER_ORDER,
]

OUTPUT_DIR = Path("results")

BERT_MODEL = "google-bert/bert-base-uncased"

MODELS_CAPABLE_OF_FINETUNING = ["relu_mlp", "resnet", "shallow_mlp", "no_activation_mlp"]
FEATURISERS_CAPABLE_OF_FINETUNING = ["bert_cls", "bert_mean"]
