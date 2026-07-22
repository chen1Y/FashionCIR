import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F


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
            for t in tqdm(test_queries):


                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:

                    visual_query = torch.stack(visual_query).float().cuda()
                    # print("test的textual_query: ", textual_query)
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
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
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
    # out = []
    # for k in [1, 10, 50]:
    #     r = 0.0
    #     for i, nns in enumerate(nn_result):
    #         if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
    #             r += 1
    #     r = 100 * r / len(nn_result)
    #     out += [('{}_r{}'.format(name, k), r)]

    out = []
    # 新增：用于记录每次检索成功时的 ref 图片 (source) ID
    successful_recalls = {1: [], 10: [], 50: []} 
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            target_id = test_queries[i]['target_img_id']
            # 判断是否成功召回了目标图片
            if test_targets_id.index(target_id) in nns[:k]:
                r += 1
                # 如果召回成功，记录当前 query 对应的 ref 图片 ID
                # ref_id = test_queries[i]['source_img_id']
                ref_id = test_queries[i]['source_img_path']
                successful_recalls[k].append(ref_id)
                
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]
    # 打印成功召回的 ref 图片名
    print(f"[{name}] K=1 成功召回的 ref 图片数: {len(successful_recalls[1])}")
    print(f"[{name}] K=1 成功召回的部分 ref 图片: {successful_recalls[1]}...") 
    # 将包含 ref 图片 ID 的结果保存到 JSON 方便后续查看：
    # import json
    # with open(f'{name}_successful_ref_recalls.json', 'w') as f:
    #     json.dump(successful_recalls, f, indent=4)

    return out



def test_cirr_valset(params, model, testset):
    
    model.eval()
    test_queries, test_targets = testset.val_queries, testset.val_targets
    with torch.no_grad():
        all_queries = []
        all_imgs = []

        if test_queries:
            # compute test query features
            visual_query = []
            textual_query = []
            for t in tqdm(test_queries):
                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:

                    visual_query = torch.stack(visual_query).float().cuda()
                    f = model.extract_query(textual_query, visual_query)
                    f = f.data.cpu().numpy()
                    all_queries += [f]

                    visual_query = []
                    textual_query = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()

                    imgs = model.extract_target(imgs).data.cpu().numpy()
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
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] # (m,n)

    # all set recalls
    cirr_out = []
    for k in [1, 5, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_r{}'.format('cirr',k), r)]

    # subset recalls
    for k in [1, 2, 3]:
        r = 0.0
        for i, nns in enumerate(nn_result):

            subset = np.array([test_targets_id.index(idx) for idx in test_queries[i]['subset_id']]) 
            subset_mask = (nns[..., None] == subset[None, ...]).sum(-1).astype(bool) 
            subset_label = nns[subset_mask] 
            if test_targets_id.index(test_queries[i]['target_img_id']) in subset_label[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_subset_r{}'.format('cirr', k), r)]

    return cirr_out



def test_fashion200k_dataset(params, model, testset):
    """Tests a model over the given testset."""
    
    model.eval()
    test_queries = testset.get_test_queries()
    with torch.no_grad():
        all_imgs = []
        all_captions = []
        all_queries = []
        all_target_captions = []
        if test_queries:
            # compute test query features
            imgs = []
     
            visual_query = []
            textual_query = []
            for t in tqdm(test_queries):
                visual_query += [testset.get_written_img(t['source_img_id'], t['target_word'])]
                textual_query += [t['source_caption'] + ', but ' + t['mod']['str']]

                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda()
                    f = model.extract_query(textual_query, visual_query).data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    visual_query = []
                    textual_query = []
            all_queries = np.concatenate(all_queries)
            all_target_captions = [t['target_caption'] for t in test_queries]

            # compute all image features
            imgs = []
            for i in tqdm(range(len(testset.imgs))):
                imgs += [testset.get_img(i)]
                if len(imgs) >= params.batch_size or i == len(testset.imgs) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)
            all_captions = [img['captions'][0] for img in testset.imgs]

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])

        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_imgs.T)
        if test_queries:
            for i, t in enumerate(test_queries):
                sims[i, t['source_img_id']] = -10e10  
        nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

        # compute recalls
        out = []
        nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

        return out