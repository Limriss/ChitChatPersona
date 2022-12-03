import torch
import numpy as np

from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

def compute_bleu_f1(args, inputs, labels):
    results = {'bleu': [],
               'f1': [],
               'recall': []}

    print(inputs.shape)

    out = args.model.generate(
        inputs,
        early_stopping=True,
        max_length=args.max_length+64,
        num_beams=5,
        no_repeat_ngram_size=3
    ).to('cpu').detach().numpy()
    
    out = list(out)

    end_of_turn_id = args.tokenizer.convert_tokens_to_ids(args.end_of_turn)
    str_preds = []
    str_labels = []
    for i in range(len(out)):
        preds = list(out[i])
        if end_of_turn_id in preds:
            preds = preds[preds.index(end_of_turn_id):]
        line = []
        for j in range(len(preds)):
            line.append(str(preds[j]))
        str_preds.append(line)
            
    for i in range(len(labels)):
        line = []
        for j in range(len(labels[i])):
            line.append(str(labels[i][j]))
        str_labels.append(line)
        while '-100' in str_labels[i]:
            str_labels[i].remove('-100')

    for pred, label in zip(str_preds, str_labels):
        if len(label) < len(pred):
            label.extend(['-100' for _ in range(len(label), len(pred))])
        elif len(label) > len(pred):
            pred.extend([str(args.tokenizer.pad_token_id) for _ in range(len(pred), len(label))])

        results['bleu'].append(bleu_score([pred], [[label]], max_n=4, weights=[0.25, 0.25, 0.25, 0.25]))
        results['f1'].append(f1_score(label, pred, average='macro'))
        results['recall'].append(recall_score(label, pred, average='macro'))
    return results
    

def eval_model(args, val_loader, skip_generation=False):
    total_loss = 0.0
    bleu = []
    f1 = []
    recall = []

    args.model.eval()

    with tqdm(total=len(val_loader), unit='batch') as tepoch:
        tepoch.set_description(f"Validating")

        for i, batch in enumerate(val_loader):
            tepoch.update(1)
            batch = {k: v.to(args.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = args.model(**batch)

            logits = outputs.logits.to('cpu')

            labels = batch.pop('labels').to('cpu').detach().numpy()

            if not skip_generation:
                results = compute_bleu_f1(args, batch['input_ids'], labels)
                bleu.extend(results['bleu'])
                f1.extend(results['f1'])
                recall.extend(results['recall'])

            loss = outputs.loss
            total_loss += loss.to('cpu').detach().numpy()
            tepoch.set_postfix({'Batch': i+1, 'Val loss (in progress)': total_loss / (i+1)})

        tepoch.set_postfix({'Val loss (final)': total_loss / len(val_loader)})
        tepoch.close()

        print(f"val ppl: {np.exp(total_loss / len(val_loader))}")
        if not skip_generation:
            print(f"val bleu: {np.mean(bleu)}")
            print(f"val f1: {np.mean(f1)}")
            print(f"val recall: {np.mean(recall)}")