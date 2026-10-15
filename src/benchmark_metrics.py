"""CIRR subset metrics computed after reference exclusion, using gallery IDs."""
import numpy as np
import torch


def cirr_metrics(similarities, queries, gallery_index, ranks):
    output = [("cirr_r5", float(np.mean(np.asarray(ranks) <= 5) * 100))]
    subset_ranks = []
    for scores, row in zip(similarities, queries):
        ids = [name for name in row["subset_id"] if name != row["source_img_id"]]
        if len(ids) != 5 or len(set(ids)) != 5 or row["target_img_id"] not in ids:
            raise ValueError("CIRR subset must contain reference and five other images including target")
        positions = {gallery_index[name] for name in ids}
        # Filter the full ranking exactly as the official evaluator does;
        # independently sorting the six members changes ties between images.
        order = [position for position in torch.argsort(scores, descending=True).tolist() if position in positions]
        subset_ranks.append(order.index(gallery_index[row["target_img_id"]]) + 1)
    output += [(f"cirr_subset_r{k}", float(np.mean(np.asarray(subset_ranks) <= k) * 100))
               for k in (1, 2, 3)]
    output.append(("cirr_standard_average", (float(np.mean(np.asarray(ranks) <= 5)*100)
                                           + output[1][1]) / 2))
    return output
