from Transformer import MyTransformer
import torch

# batch_size = 2
# src_len = 5
# tgt_len = 6
# vocab_size = 20

# # toy input
# src = torch.randint(0, vocab_size, (batch_size, src_len))
# tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

# print("input shape:", src.shape)

# model = MyTransformer(token_num=vocab_size)

# out = model(src, tgt)

# print("Output shape:", out.shape)

import torch
from torch import nn

# model
model = MyTransformer()
# loss function
loss_fn = nn.CrossEntropyLoss()
# learning rate and optimizer
def get_lr(step, warm_up_step, d_model):
    return (d_model ** -0.5 * min(step ** -0.5, step * warm_up_step ** -1.5))

learning_rate = get_lr(step, warm_up_step, d_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98) eps=10e-9)


# setup parameters
total_train_step = 0   # number of training
total_test_step = 0  # number of testing
epoch = 20     # role of training

# tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("../logs_transformer_train")

# training
for i in range(epoch):
    print(f"------- Epoch {i+1} Training Started -------")

    # training start
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)
        
        # backward propogation and updating parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"number of training: {total_train_step}, loss: {loss.item()}") #.item() changes tensor to a number
            writer.add_scalar("train_loss", loss.item(), total_train_step) # record loss in tensorboard
    
    # testing start (after each epoch)
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():   # no gradient calculation
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"loss on the entire test set: {total_test_loss}")
    print(f"accuracy on the entire test set: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss.item(), total_test_step) # record loss in tensorboard
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # save each epoch
    torch.save(model, f"Transformer_model_{i}.pth")
    print("model saved successfully")

writer.close()
    