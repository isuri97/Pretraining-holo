from transformers import pipeline

text = "<mask> were taken into the <mask> <mask>."

mask_filler = pipeline("fill-mask", "/tmp/test-mlm")
print(mask_filler(text, top_k=3))