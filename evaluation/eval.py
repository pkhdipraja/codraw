import numpy as np
import torch.nn as nn
import datasets.codraw_data as codraw_data
import torch

from datasets.episode import respond_to, Episode
from datasets.codraw_data import Clipart

@respond_to(codraw_data.ObserveTruth)
@respond_to(codraw_data.ReplyGroup)
def scripted_tell(episode):
    if episode.script_index < len(episode.script):
        event = episode.script[episode.script_index]
        if isinstance(event, codraw_data.Peek):
            # Skip to the next non-peek event
            assert isinstance(episode.script[episode.script_index + 1], codraw_data.TellerObserveCanvas)
            episode.script_index += 2
            return scripted_tell(episode)
        episode.script_index += 1
        episode.append(event)


@respond_to(codraw_data.ObserveTruth)
@respond_to(codraw_data.ReplyGroup)
def scripted_tell_before_peek(episode):
    if episode.script_index < len(episode.script):
        event = episode.script[episode.script_index]
        if isinstance(event, codraw_data.Peek):
            return
        episode.script_index += 1
        episode.append(event)


@respond_to(codraw_data.ObserveTruth)
@respond_to(codraw_data.ReplyGroup)
def scripted_tell_after_peek(episode):
    if episode.script_index == 0:
        while episode.script_index < len(episode.script):
            event = episode.script[episode.script_index]
            episode.script_index += 1
            if not isinstance(event, codraw_data.Peek):
                continue
            event = episode.script[episode.script_index]
            assert isinstance(event, codraw_data.TellerObserveCanvas)
            start_scene = event.scene
            episode.script_index += 1
            break
        else:
            assert False, "Could not find Peek event in the script!"
        episode.append(codraw_data.DrawGroup(start_scene))
        assert episode.script_index < len(episode.script)

    if episode.script_index < len(episode.script):
        event = episode.script[episode.script_index]
        episode.script_index += 1
        episode.append(event)


def make_fns(splits, *objs_or_pairs):
    split_to_use = 0
    res = []
    for obj_or_pair in objs_or_pairs:
        if isinstance(obj_or_pair, tuple):
            assert len(obj_or_pair) == 2
            if splits[split_to_use] == 'a':
                obj = obj_or_pair[0]
            elif splits[split_to_use] == 'b':
                obj = obj_or_pair[1]
            else:
                raise ValueError("Invalid split: {}".format(splits[split_to_use]))
            split_to_use += 1
        else:
            obj = obj_or_pair
        
        # if isinstance(obj, nn.Module):
        #     obj.eval()
        
        if hasattr(obj, 'get_action_fns'):
            res.extend(obj.get_action_fns())
        else:
            res.append(obj)
    
    assert split_to_use == len(splits), "Too many splits specified"
    return res


def episodes_from_fns(cfgs, fns, limit=None, split='dev'):
    use_scripts = (scripted_tell in fns) or (scripted_tell_before_peek in fns)
    if scripted_tell_after_peek in fns:
        use_scripts = True
        run_from = codraw_data.get_scenes_and_scripts_with_peek(cfgs, split)
    elif use_scripts:
        run_from = codraw_data.get_scenes_and_scripts(cfgs, split)
    else:
        run_from = codraw_data.get_scenes(cfgs, split)

    if limit is not None:
        run_from = run_from[:limit]

    sims = []
    with torch.no_grad():
        for run_from_single in run_from:
            if use_scripts:
                episode = Episode.run_script(run_from_single, fns)
            else:
                episode = Episode.run(run_from_single, fns)
            yield episode


def eval_fns(cfgs, fns, limit=None, split='dev'):
    sims = [episode.scene_similarity() for episode in episodes_from_fns(cfgs, fns, limit=limit, split=split)]
    return np.array(sims)


class ComponentEvaluator(object):
    NUM_FEATURES = 7

    _instance_cache = {}

    @classmethod
    def get(cls, cfgs, split_for_baseline='train_full'):
        if split_for_baseline not in cls._instance_cache:
            cls._instance_cache[split_for_baseline] = cls(cfgs, split_for_baseline)
        return cls._instance_cache[split_for_baseline]

    def __init__(self, cfgs, split_for_baseline='train_full'):
        self.cfgs = cfgs
        cliparts_by_idx = {idx: [] for idx in range(58)}
        for scene in codraw_data.get_scenes(cfgs, split_for_baseline):
            for clipart in scene:
                cliparts_by_idx[clipart.idx].append(clipart)
        
        self.idx_to_exemplar = {}
        for idx in cliparts_by_idx:
            if idx in Clipart.HUMAN_IDXS:
                expression, _ = torch.mode(torch.tensor([c.expression for c in cliparts_by_idx[idx]]))
                pose, _ = torch.mode(torch.tensor([c.pose for c in cliparts_by_idx[idx]]))
                subtype = pose * Clipart.NUM_EXPRESSION + expression
            else:
                subtype = 0
            depth, _ = torch.mode(torch.tensor([c.depth for c in cliparts_by_idx[idx]]))
            flip, _ = torch.mode(torch.tensor([c.flip for c in cliparts_by_idx[idx]]))

            x = np.mean([c.x for c in cliparts_by_idx[idx]])
            y = np.mean([c.y for c in cliparts_by_idx[idx]])
            self.idx_to_exemplar[idx] = Clipart(idx, int(subtype), int(depth), int(flip), x, y)
        
        # Calculate prior baseline, and human performance
        human_numer = np.zeros(self.NUM_FEATURES)
        human_denom = np.zeros(self.NUM_FEATURES)
        baseline_numer = np.zeros(self.NUM_FEATURES)
        baseline_denom = np.zeros(self.NUM_FEATURES)
        for scene_true, scene_human in codraw_data.get_truth_and_human_scenes(cfgs, 'dev'):
            ep_numer, ep_denom = self.eval_scene(scene_human, scene_true)
            human_numer += ep_numer
            human_denom += ep_denom
            ep_numer, ep_denom = self.eval_scene([], scene_true)
            baseline_numer += ep_numer
            baseline_denom += ep_denom

        self.human_scores = human_numer / human_denom
        self.baseline_scores = baseline_numer / baseline_denom
    
    def eval_scene(self, pred, target):
        res_numer = np.zeros(self.NUM_FEATURES)
        res_denom = np.zeros(self.NUM_FEATURES)

        for truth_clipart in target:
            other_cliparts = [c for c in pred if c.idx == truth_clipart.idx]
            if other_cliparts:
                other_clipart = other_cliparts[0]
            else:
                other_clipart = self.idx_to_exemplar[truth_clipart.idx]

            feats_numer = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            feats_denom = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            feats_numer[0] = float(truth_clipart.flip != other_clipart.flip)
            if truth_clipart.idx in Clipart.HUMAN_IDXS:
                feats_numer[1] = float(truth_clipart.expression != other_clipart.expression)
                feats_numer[2] = float(truth_clipart.pose != other_clipart.pose)
                feats_denom[1] = 1.0
                feats_denom[2] = 1.0
            feats_numer[3] = float(truth_clipart.depth != other_clipart.depth)
            displacements = np.array([truth_clipart.normed_x - other_clipart.normed_x, truth_clipart.normed_y - other_clipart.normed_y])
            feats_numer[4] = np.sum(displacements ** 2)
            feats_numer[5], feats_numer[6] = np.abs(displacements)

            res_numer += feats_numer
            res_denom += feats_denom
        return res_numer, res_denom
    
    def eval_episode(self, episode):
        return self.eval_scene(episode.reconstruct(), episode.get_true_scene())
    
    def eval_fns(self, fns, limit=None, split='dev', unscaled=False):
        numer = np.zeros(self.NUM_FEATURES)
        denom = np.zeros(self.NUM_FEATURES)
        for episode in episodes_from_fns(self.cfgs, fns, limit=limit, split=split):
            ep_numer, ep_denom = self.eval_episode(episode)
            numer += ep_numer
            denom += ep_denom

        res = numer / denom
        if not unscaled:
            res = (res - self.human_scores) / (self.baseline_scores - self.human_scores)
            res = 1.0 - res

        return res