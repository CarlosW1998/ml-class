from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./token",
    tokenizer="./model"
)

fill_mask("La suno <mask>.")