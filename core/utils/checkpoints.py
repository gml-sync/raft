import json
from pathlib import Path
import torch


def read_json(filename):
    data_dict = None
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data_dict = json.load(f)
        except:
            pass

    return data_dict


def write_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)


def checkpoint_load_path(path):
    '''
    path - str. Path of checkpoint
    returns path or backup path based on saved json.
    '''
    p = Path(path)
    info = Path(p.parent / (p.stem + '.json'))
    if not info.exists():
        return path

    checkpoint_info = read_json(info)
    if checkpoint_info is None: # JSON corruption means both checkpoints are good
        return path

    if checkpoint_info['backup_latest']:
        return p.parent / (p.stem + '.back')
    else:
        return path


def checkpoint_save_path(path, save_json=False):
    '''
    path - str. Path of checkpoint
    returns path or backup path based on saved json.
        JSON save must be done AFTER checkpoint saving
    '''
    p = Path(path)
    save_to_back = False

    info = Path(p.parent / (p.stem + '.json'))
    if info.exists():
        checkpoint_info = read_json(info)
        if not checkpoint_info is None:
            # JSON corruption means both checkpoints are good. Assume JSON is fine
            if not checkpoint_info['backup_latest']:
                # Latest checkpoint is not backup
                save_to_back = True

    if save_json:
        write_json({'backup_latest': save_to_back}, info)

    if not save_to_back:
        return path
    else:
        return p.parent / (p.stem + '.back')


def save_model_txt(model, path):
    fout = open(path, 'w')
    for k, v in model.state_dict().items():
        fout.write(str(k) + '\n')
        fout.write(str(v.tolist()) + '\n')
    fout.close()

def load_model_txt(model, path):
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        print('Iter', i)
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            data_dict[s] = 0
            prev_key = s
        else:
            val = eval(s)
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
        odd = (odd + 1) % 2
        i += 1

    # Replace existing values with loaded

    print('Loading...')
    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                sys.exit(0)
    print('Model loaded')

def convert_to_txt(model, model_path, save_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    save_model_txt(model, save_path)