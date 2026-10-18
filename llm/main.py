from dataloader import create_dataloader

with open("../pretraining-data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()



dl = create_dataloader(
    raw_text, 
    batch_size=1, 
    max_length=4, 
    stride=1, 
    shuffle=True, 
    drop_last=True, 
    num_workers=0
    )


dl_iter = iter(dl)
first_batch = next(dl_iter)
print("First batch input IDs:", first_batch)