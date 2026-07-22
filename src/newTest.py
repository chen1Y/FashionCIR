import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast

def test(params, model, testset, category):
    # 切换到eval模式
    model.eval()
    (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            visual_query = []
            textual_query = []
            # 提取查询特征
            for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda()
                    with autocast():
                        f = model.extract_query(textual_query, visual_query)
                    f = f.data.cpu().numpy()
                    all_queries += [f]

                    visual_query = []
                    textual_query = []

            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            logits = []
            # 提取目标图像特征
            for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    with autocast():
                        imgs = model.extract_target(imgs)
                    imgs = imgs.data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    
    if name != 'birds':
        for i, t in enumerate(test_queries):
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10

    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    # 计算召回率
    out = []
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]

    return out