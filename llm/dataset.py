import torch
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenzier, max_lenght, stride):
        self.input_ids = []
        self.target_id = []


        tokens = tokenzier.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tokens) - max_lenght, stride):
            temp_input_id =  tokens[i:i + max_lenght]
            temp_target_id = tokens[i + 1:i + max_lenght + 1]

            self.input_ids.append(torch.tensor(temp_input_id))
            self.target_id.append(torch.tensor(temp_target_id))

    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_id[idx]