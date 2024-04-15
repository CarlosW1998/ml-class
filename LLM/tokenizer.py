from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./datasets").glob("**/medical.txt.data")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=1000, min_frequency=2, special_tokens=[
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
])

tokenizer.save_model("./vocab/")