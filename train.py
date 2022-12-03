from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler

import torch
from loaders import get_dataloaders
from eval import eval_model
from tqdm import tqdm

import argparse, os
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description = "Decoder only transformer")

parser.add_argument('--model_name', type=str, default='gpt3-small', help='Model name or name in huggingface hub')

parser.add_argument('--batch_size',  type=int,   default=1)
parser.add_argument('--num_workers', type=int,   default=0)
parser.add_argument('--max_length',  type=int,   default=1024)
parser.add_argument('--split_size',  type=float, default=0.5)
parser.add_argument('--split_data',  type=str,   default='toloka', help='Can be \'both\', \'personachat\' or \'toloka\'')

parser.add_argument('--num_epochs',         type=int,   default=3)
parser.add_argument('--lr',                 type=float, default=5e-5)
parser.add_argument('--weight_decay',       type=float, default=1e-4)
parser.add_argument('--num_warmup_steps',   type=int,   default=1500)
parser.add_argument('--accumulation_steps', type=int,   default=64)

args = parser.parse_args()

model_path = os.path.join(os.curdir, 'models', args.model_name)
data_path   = os.path.join(os.curdir, 'data')

person1_token  = '<ps1>'
person2_token  = '<ps2>'
profile1_token = '<pi1>'
profile2_token = '<pi2>'
end_of_turn    = '<end>'

ATTR_TO_SPECIAL_TOKEN = {
    'additional_special_tokens': [
        person1_token, person2_token, profile1_token, profile2_token, end_of_turn
    ]
}

personachat_path = os.path.join(data_path, "train_both_original.txt")
toloka_path      = os.path.join(data_path, "toloka_speller.txt")

save_path = f"{model_path}-trained"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, truncation_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path, max_length=args.max_length).to(device)

if 'gpt' in args.model_name:
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.pad_token_id

num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)

model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + num_added_tokens + 1)

args.tokenizer        = tokenizer
args.model            = model
args.device           = device
args.person1_token    = person1_token
args.person2_token    = person2_token
args.profile1_token   = profile1_token
args.profile2_token   = profile2_token
args.end_of_turn      = end_of_turn
args.personachat_path = personachat_path
args.toloka_path      = toloka_path

train_loader, val_loader = get_dataloaders(args, val=args.split_data, split_size=args.split_size)

num_training_steps = args.num_epochs * len(train_loader) // args.accumulation_steps

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_training_steps
    )

ppl = []

scaler = torch.cuda.amp.GradScaler()

for epoch in range(args.num_epochs):
    print(f"Epoch: {epoch+1}")
    print("Training")
    optimizer.zero_grad()
    total_loss = 0.0
    
    model.train()

    with tqdm(total=len(val_loader), unit='batch') as tepoch:
        tepoch.set_description(f"Training. Epoch: {epoch+1}")

        for i, batch in enumerate(train_loader):
            tepoch.update(1)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**batch)

            loss = outputs.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            
            if i % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += outputs.loss.to('cpu').detach().numpy()
            tepoch.set_postfix({'Batch': i+1, 'Train loss (in progress)': total_loss / (i+1)})

        tepoch.set_postfix({'Train loss (final)': total_loss / len(train_loader)})
        tepoch.close()
                
    eval_model(args, val_loader, skip_generation=True)