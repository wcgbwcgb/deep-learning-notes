from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("D:/deep-learning-notes/Attention_is_All_You_Need/test_logs")
print("start")
writer.add_scalar("test", 1, 1)
print("success")
writer.close()