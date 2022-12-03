import torch
from dataset import TextDataset, collate_fn

#PersonaChat data loading
def load_personachat(args):
    context = []
    label   = []
    persona = []

    person1_token  = args.person1_token
    person2_token  = args.person2_token
    profile1_token = args.profile1_token
    profile2_token = args.profile2_token
    end_of_turn    = args.end_of_turn

    with open(args.personachat_path, 'r') as infile:
        lines = infile.readlines()

        prev = 0
        session_persona = ""
        session_context = ""
        for line in lines:
            index = line.find(' ')

            if int(line[:index]) < prev:
                session_persona = session_persona.replace('\n', '')
                session_context = session_context.replace('\n', '\t').split('\t')
                session_context.pop()

                token = person2_token
                for i in range(len(session_context)):
                    session_context[i] = token + session_context[i]
                    if token == person1_token:
                        token = person2_token
                    else:
                        token = person1_token

                for j in range(0, len(session_context), 2):
                    persona.append(session_persona)
                    dialogue = ''.join(session_context[:j+2])
                    label.append(session_context[j+1] + end_of_turn)
                    context.append(dialogue)

                session_persona = ""
                session_context = ""

            prev = int(line[:index])
            if 'your persona:' in line:
                start = line.find('your persona:') + 14
                session_persona += profile1_token + line[start:]
            elif 'partner\'s persona:' in line:
                start = line.find('partner\'s persona:') + 19
                session_persona += profile2_token + line[start:]
            else:
                session_context += line[index+1:line.find('\t\t')] + '\n'


    return {
        "persona": persona,
        "context": context,
        "labels": label
        }

#Toloka PersonaChat Rus data loading
def load_toloka(args):
    context = []
    persona = []
    label = []

    person1_token  = args.person1_token
    person2_token  = args.person2_token
    profile1_token = args.profile1_token
    profile2_token = args.profile2_token
    end_of_turn    = args.end_of_turn

    with open(args.toloka_path, 'r', encoding="utf-8") as infile:
        lines = infile.readlines()
        lines = [line.rstrip() for line in lines]

        for i in range(0, len(lines), 3):
            persona1 = lines[i].replace('\n', '').replace('|', profile1_token).replace('.', '')
            persona2 = lines[i+1].replace('\n', '').replace('|', profile2_token).replace('.', '')
            
            session_context = lines[i+2]
            
            p1 = session_context.find("Пользователь 1: ")
            p2 = session_context.find("Пользователь 2: ")
            
            while p1 >= 0 and p2 >= 0:
                if p1 < p2:
                    session_context = session_context.replace("Пользователь 1: ", f"\n{person1_token}", 1)
                    while session_context.find("Пользователь 1: ") < p2 and session_context.find("Пользователь 1: ") > -1:
                        session_context = session_context.replace("Пользователь 1: ", f" ", 1)
                    p1 = session_context.find("Пользователь 1: ")
                else:
                    session_context = session_context.replace("Пользователь 2: ", f"\n{person2_token}", 1)
                    while session_context.find("Пользователь 2: ") < p1 and session_context.find("Пользователь 2: ") > -1:
                        session_context = session_context.replace("Пользователь 2: ", f" ", 1)
                    p2 = session_context.find("Пользователь 2: ")
                    
            if p1 > -1:
                session_context = session_context.replace("Пользователь 1: ", f"\n{person1_token}", 1)
                while session_context.find("Пользователь 1: ") > -1:
                    session_context = session_context.replace("Пользователь 1: ", f" ", 1)
            elif p2 > -1:
                session_context = session_context.replace("Пользователь 2: ", f"\n{person2_token}", 1)
                while session_context.find("Пользователь 2: ") > -1:
                    session_context = session_context.replace("Пользователь 2: ", f" ", 1)
            
            session_context = session_context.replace(")", "").replace("(", "").split("\n")
            session_context.pop(0)        

            session_persona = persona1 + persona2
            
            for j in range(0, len(session_context) - len(session_context) % 2, 2):
                persona.append(session_persona)
                dialogue = ''.join(session_context[:j+2])
                label.append(session_context[j+1] + end_of_turn)
                context.append(dialogue)

    return {
        "persona": persona,
        "context": context,
        "labels": label
        } 


def get_dataloaders(args, val='toloka', split_size=0.5):
    personachat_data = load_personachat(args)
    toloka_data      = load_toloka(args)

    dataset_size = len(personachat_data['persona']) + len(toloka_data['persona'])

    dataset  = None
    to_split = None

    train_size = 0
    eval_size  = 0

    train_data = None
    val_data   = None
    #Defining how and what to split
    if val == 'both':
        train_size = int(split_size * dataset_size)
        eval_size = dataset_size - train_size
        to_split = TextDataset({
            "persona": personachat_data['persona'] + toloka_data['persona'],
            "context": personachat_data['context'] + toloka_data['context'],
            "labels" : personachat_data['labels'] + toloka_data['labels']
        })
    elif val == 'personachat':
        train_size = int(split_size * len(personachat_data['persona']))
        eval_size = len(personachat_data['persona']) - train_size
        to_split = TextDataset({
            "persona": personachat_data['persona'],
            "context": personachat_data['context'],
            "labels" : personachat_data['labels']
        })
        dataset = TextDataset({
            "persona": toloka_data['persona'],
            "context": toloka_data['context'],
            "labels" : toloka_data['labels']
        })
    elif val == 'toloka':
        train_size = int(split_size * len(toloka_data['persona']))
        eval_size = len(toloka_data['persona']) - train_size
        to_split = TextDataset({
            "persona": toloka_data['persona'],
            "context": toloka_data['context'],
            "labels" : toloka_data['labels']
        })
        dataset = TextDataset({
            "persona": personachat_data['persona'],
            "context": personachat_data['context'],
            "labels" : personachat_data['labels']
        })
    else:
        raise ValueError('Wrong value for variable val. Use \'both\', \'personachat\' or \'toloka\'')

    train_data, val_data = torch.utils.data.random_split(
            to_split, 
            [train_size, eval_size], 
            generator=torch.Generator().manual_seed(42)
        )

    if dataset != None:
        train_data = torch.utils.data.ConcatDataset([train_data, dataset])

    return (
        torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            collate_fn=lambda data=train_data, tokenizer=args.tokenizer, max_length=args.max_length: collate_fn(data, tokenizer, max_length)),
        torch.utils.data.DataLoader(
            val_data, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=lambda data=val_data, tokenizer=args.tokenizer, max_length=args.max_length: collate_fn(data, tokenizer, max_length))
    )