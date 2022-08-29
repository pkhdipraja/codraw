import numpy as np


def scene_similarity(pred, target):
    """
    This version of the scene similarity metric should be monotonic, in the
    sense that adding correct clipart should always increase the score, adding
    incorrect clipart should decrease it, and removing incorrect clipart should
    increase it. It also breaks out the different components of Mike/Jenny:
    flip, expression, and pose; as well as capping distance error at 1.
    """
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)
    iou = len(idx1 & idx2) / len(idx1 | idx2)

    intersection_size = len(idx1 & idx2)
    union_size = len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(8)
    denom = np.zeros(8)

    num[0] = intersection_size

    for c1, c2 in zip(match1, match2):
        num[1] += int(c1.flip != c2.flip)
        if c1.idx in c1.HUMAN_IDXS:
            num[2] += int(c1.expression != c2.expression)
            num[3] += int(c1.pose != c2.pose)
        num[4] += int(c1.depth != c2.depth)
        num[5] += min(1.0, np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2))
    
    denom[:6] = union_size

    for idx_i in range(len(match1)):
        for idx_j in range(idx_i, len(match1)):
            if idx_i == idx_j:
                continue
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # TODO(nikita): this doesn't correctly handle the case if two
            # cliparts have *exactly* the same x/y coordinates in the target
            num[6] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) <= 0)
            num[7] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) <= 0)

    denom[6:] = union_size * (intersection_size - 1)

    denom = np.maximum(denom, 1)

    score_components = num / denom
    score_weights = np.array([5,-1,-0.5,-0.5,-1,-1,-1,-1])

    return score_components @ score_weights