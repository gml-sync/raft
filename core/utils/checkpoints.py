import json
from pathlib import Path


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
    info = Path(p.stem + '.json')
    if not info.exists():
        return path

    checkpoint_info = read_json(info)
    if checkpoint_info is None: # JSON corruption means both checkpoints are good
        return path

    if checkpoint_info['backup_latest']:
        return path + '.back'
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

    info = Path(p.stem + '.json')
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
        return path + '.back'


def save_model_crossplatform(model, path):
    fout = open(path, 'w')
    for k, v in model.state_dict().items():
        fout.write(str(k) + '\n')
        fout.write(str(v.tolist()) + '\n')
    fout.close()

def load_model_crossplatform(path):
    data_dict = {}
    fin = open(path, 'r')
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            data_dict[s] = 0
            prev_key = s
        else:
            data_dict[prev_key] = torch.FloatTensor(eval(s))
        odd = (odd + 1) % 2
    return data_dict