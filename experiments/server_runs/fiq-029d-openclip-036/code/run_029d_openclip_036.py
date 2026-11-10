"""Initialize 029D from public OpenCLIP and compare with the sealed 035 control."""

import argparse
import gc
import os
import time
from pathlib import Path

import open_clip
import torch

from experiment_storage import require_space
from fiq_fusion_029 import FusionText029
from fiq_native_caption_math_031 import metrics, transitions
from run_029d_freeze_patch import eval_global, ranks_for
from run_029d_full_clip_034 import CATEGORIES, official_bank
from run_029d_gate_refresh import PARENT, inputs_029
from run_029d_inbatch_035 import assert_card
from run_attr_transfer_029d import ARM, encode_bank, fit, training_context
from run_fiq_native_caption_031 import identity, read, sha, write


OPENCLIP_SHA = '9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4'
OPENCLIP_SIZE = 3944692325


def training_meta(category):
    if category != 'dress':
        return inputs_029(category, 'train')[0]
    path = PARENT/'dress/train.json'
    if sha(path) != read(PARENT/'launch.json')['inputs']['dress']:
        raise ValueError('Dress training metadata differs from the 029D source')
    return read(path)


def from_public_clip(path):
    if path.stat().st_size != OPENCLIP_SIZE or sha(path) != OPENCLIP_SHA:
        raise ValueError('Published OpenCLIP checkpoint checksum mismatch')
    clip, _, transform = open_clip.create_model_and_transforms('ViT-H-14', pretrained=None)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    weights = checkpoint.get('state_dict', checkpoint)
    clip.load_state_dict({name.removeprefix('module.'):value for name,value in weights.items()}, strict=True)
    del weights, checkpoint
    clip.requires_grad_(False)
    clip.eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model = FusionText029(clip, tokenizer, 2).cuda()
    model.configure_arm(ARM)
    return model, transform


def run(root, category, gpu):
    directory = root/category
    directory.mkdir()
    status = directory/'state.json'
    state = dict(status='running',category=category,identity=identity(),gpu=gpu,api_calls=0)
    started = time.monotonic()

    def progress(stage, done=0, total=0):
        if (root/'cancel.request').exists() or time.monotonic()-started > 7200:
            raise InterruptedError('Experiment cancelled or timed out')
        assert_card(gpu)
        state.update(stage=stage,done=done,total=total)
        write(status,state)

    try:
        assert_card(gpu,startup=True)
        require_space(root,20)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
            raise RuntimeError('GPU UUID was not bound')
        plan = read(root/'plan.json')
        for name, digest in plan['source_sha256'].items():
            if sha(root/'code'/name) != digest:
                raise ValueError(f'Sealed source changed: {name}')
        control_path = root.parent/'fiq-029d-inbatch-035'/category/'results.json'
        if sha(control_path) != plan['control_sha256'][category]:
            raise ValueError('035 control evidence checksum mismatch')
        control = read(control_path)
        progress('public_clip_load')
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.65)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, transform = from_public_clip(root/'weights/open_clip_pytorch_model.bin')
        meta = training_meta(category)
        source_rows = meta['records']
        fit_n,dev_n = (len(meta['offsets'][s]) for s in ('fit','dev'))
        if (fit_n != len(control['fits']['full_b8']['exposures'])
                or dev_n != len(control['fits']['full_b8']['history'][0]['metrics']['ranks'])):
            raise ValueError('036 training or dev denominator differs from 035')
        ids = list(dict.fromkeys(meta['banks']['fit']+meta['banks']['dev']))
        train_bank = encode_bank(model.clip,transform,model.tokenizer,category,'train',source_rows,ids,
                                 meta['image_sha256'],progress)
        context = training_context(train_bank,meta)
        del train_bank
        torch.manual_seed(42)
        choice = fit(directory,model,context,progress)
        fitted = read(directory/'fit_results.json')
        if (fitted['exposures'] != control['fits']['full_b8']['exposures']
                or fitted['updates'] != control['fits']['full_b8']['updates']):
            raise ValueError('Training exposure differs from 035')
        del context
        gc.collect()
        torch.cuda.empty_cache()
        weight = directory/choice['weight']
        if sha(weight) != choice['sha256']:
            raise ValueError('Selected 036 weight changed')
        model.load_delta(torch.load(weight,map_location='cpu',weights_only=True)['state'])
        model.eval()
        val, hashes, native = official_bank(category)
        rows = [dict(row,native_text=text) for row,text in zip(val['rows'],native)]
        if (len(rows),len(val['ids'])) != (control['queries'],control['gallery']):
            raise ValueError('Official gallery/query denominator differs from 035')
        bank = encode_bank(model.clip,transform,model.tokenizer,category,'val',rows,val['ids'],hashes,progress)
        scores, _ = eval_global(model,bank,False,lambda d,t:progress('official_ranks',d,t))
        output = {}
        comparisons = {}
        for view, value in scores.items():
            ranks = ranks_for(value,bank)
            name = f'public_clip_{view}'
            output[name] = dict(metrics(ranks),ranks=ranks)
            control_name = f'dqu_init_{view}'
            output[control_name] = control['rows'][f'full_b8_{view}']
            comparisons[name+'_minus_'+control_name] = {str(k):transitions(ranks,output[control_name]['ranks'],k)
                                                         for k in (1,10,50)}
        output['native_dqu'] = control['rows']['dqu']
        write(directory/'results.json',dict(category=category,queries=len(rows),gallery=len(val['ids']),rows=output,
              fit=fitted,comparisons=comparisons,openclip_sha256=OPENCLIP_SHA,control_sha256=sha(control_path),
              api_calls=0,vision_trainable=False,fusion_initialization='random'))
        state.update(status='complete',stage='complete',seconds=time.monotonic()-started)
        write(status,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__,detail=str(error))
        write(status,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--category',choices=CATEGORIES,required=True)
    parser.add_argument('--gpu',required=True)
    args = parser.parse_args()
    run(args.root,args.category,args.gpu)
