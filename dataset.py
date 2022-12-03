import torch

def collate_fn(data, tokenizer, max_length):
    persona, context, labels = zip(*data)
    
    input_ids = []
    attention_mask = []
    answers = []
        
    for i, j, label in zip(persona, context, labels):
        tokenized_persona = tokenizer(
            i,
            max_length=max_length,
            padding=False,
            truncation=True,)
        per_len = len(tokenized_persona['input_ids'])

        tokenized_context = tokenizer(
            j, 
            max_length=max(0, max_length - per_len), 
            padding=False, 
            truncation=True)
        con_len = len(tokenized_context['input_ids'])

        tokenized_label = tokenizer(
            label,
            max_length=max_length,
            padding=False,
            truncation=True
        )
        lab_len = len(tokenized_label['input_ids'])

        input_ids.append(
            tokenized_persona['input_ids'] + \
            tokenized_context['input_ids'] + \
            [tokenizer.pad_token_id for _ in range(max(0, max_length - per_len - con_len))])
        attention_mask.append(
            tokenized_persona['attention_mask'] + \
            tokenized_context['attention_mask'] + \
            [tokenizer.pad_token_id for _ in range(max(0, max_length - per_len - con_len))])
        answers.append(
            [-100 for _ in range(min(max_length, max(0, per_len + con_len - lab_len)))] + \
            tokenized_label['input_ids'] + \
            [-100 for _ in range(min(max_length, max(0, max_length - per_len - con_len)))])
        
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.Tensor(attention_mask)
    answers = torch.LongTensor(answers)
    

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": answers
    }

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.persona = list(data['persona'])
        self.context = list(data['context'])
        self.labels = list(data['labels'])
        
    def __len__(self):
        return len(self.context)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.persona[idx], self.context[idx], self.labels[idx]
        else:
            return self.persona[idx], self.context[idx], _