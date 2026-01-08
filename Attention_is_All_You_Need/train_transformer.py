from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# load dataset
dataset = load_dataset("lilferrit/wmt14-short")
print("loaded successfully")

train_data = dataset["train"]
val_data   = dataset["validation"]
test_data  = dataset["test"]

print(len(train_data), len(val_data), len(test_data))

print(train_data[0])

# tokenize
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

def preprocess(batch):
    inputs = [ex["de"] for ex in batch["translation"]]
    targets = [ex["en"] for ex in batch["translation"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess, batched=True, cache_file_name="tokenized_dataset.arrow")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "decoder_attention_mask"])

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
test_dataloader  = DataLoader(tokenized_datasets["test"], batch_size=32)


import torch
from torch.optim import Adam
from Transformer import MyTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyTransformer(token_num=tokenizer.vocab_size,d_model=512).to(device)

optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# setup parameters
total_train_step = 0   # number of training
total_test_step = 0  # number of testing
epoch = 2     # role of training

# tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("D:/deep-learning-notes/Attention_is_All_You_Need/transformer_logs")

# training
for i in range(epoch):
    print(f"------- Epoch {i+1} Training Started -------")

    # training start
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}

        # shift right
        decoder_input_ids = torch.zeros_like(batch["labels"])
        decoder_input_ids[:, 1:] = batch["labels"][:, :-1]
        bos_token_id = tokenizer.pad_token_id  # 或 tokenizer.cls_token_id
        decoder_input_ids[:, 0] = bos_token_id   

        optimizer.zero_grad()
        output = model(batch["input_ids"], decoder_input_ids, src_attn_mask=batch["attention_mask"],trg_attn_mask=batch["decoder_attention_mask"])  # forward
        # reshape for CrossEntropyLoss: [batch*seq_len, vocab_size] vs [batch*seq_len]
        loss = loss_fn(output.view(-1, tokenizer.vocab_size), batch["labels"].view(-1))
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"number of training: {total_train_step}, loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing start (after each epoch)
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:

            # shift right
            decoder_input_ids = torch.zeros_like(batch["labels"])
            decoder_input_ids[:, 1:] = batch["labels"][:, :-1]
            bos_token_id = tokenizer.pad_token_id  # 或 tokenizer.cls_token_id
            decoder_input_ids[:, 0] = bos_token_id   

            batch = {k: v.to(device) for k,v in batch.items()}
            output = model(batch["input_ids"], decoder_input_ids, src_attn_mask=batch["attention_mask"], trg_attn_mask=batch["decoder_attention_mask"])
            loss = loss_fn(output.view(-1, tokenizer.vocab_size), batch["labels"].view(-1))
            total_test_loss += loss.item()

            pred = output.argmax(-1)
            mask = batch["labels"] != tokenizer.pad_token_id
            total_correct += (pred[mask] == batch["labels"][mask]).sum().item()
            total_tokens += mask.sum().item()

    test_accuracy = total_correct / total_tokens
    print(f"loss on the entire test set: {total_test_loss}")
    print(f"accuracy on the entire test set: {test_accuracy}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # save model
    torch.save(model.state_dict(), f"transformer_epoch_{i}.pth")
    print("model saved successfully")

writer.close()
    