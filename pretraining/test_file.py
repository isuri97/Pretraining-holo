from transformers import pipeline

text = "People were taken into the <mask> concentration camp."

mask_filler = pipeline("fill-mask", "/tmp/test-mlm")
print(mask_filler(text, top_k=3))