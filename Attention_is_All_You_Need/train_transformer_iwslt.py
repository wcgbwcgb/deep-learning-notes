from datasets import load_dataset
print("start")

# load dataset
dataset = load_dataset("google/wmt24pp", "en-zh_CN")

print(dataset["train"][0])

