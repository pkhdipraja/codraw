import torch.nn as nn
import datasets.codraw_data as codraw_data

from datasets.episode import Episode, respond_to


class Model(object):
    datagen_cls = None
    def __init__(self, cfgs, datagen=None, spec=None, **kwargs):
        super().__init__()
        self.cfgs = cfgs
        if spec is not None:
            assert self.datagen_cls is not None
            assert self.datagen_cls.__name__ == spec['datagen_class']
            self.datagen = self.datagen_cls(cfgs, spec=spec['datagen_spec'])
            self.init_from_spec(**{k: v for (k,v) in spec.items() if k not in ['class', 'datagen_spec', 'datagen_class', 'state_dict']})
            # if 'state_dict' in spec:
            #     self.load_state_dict(spec['state_dict'])
            #     self.to(cuda_if_available)
            self.post_init_from_spec()
        else:
            assert isinstance(datagen, self.datagen_cls)
            self.datagen = datagen
            self.init_full(**kwargs)
            # if hasattr(self, 'state_dict'):
            #     self.to(cuda_if_available)
    
    def init_full(self):
        pass
    
    def init_from_spec(self, **kwargs):
        self.init_full(**kwargs)
    
    def post_init_from_spec(self):
        pass
    
    def get_action_fns(self):
        raise NotImplementedError("Subclasses should override this")
    
    def get_spec(self):
        return {}
    
    @property
    def spec(self):
        res = {
            'class': type(self).__name__,
            'datagen_class': type(self.datagen).__name__,
            'datagen_spec': self.datagen.spec,
            **self.get_spec(),
        }
        if hasattr(self, 'state_dict'):
            res['state_dict'] = self.state_dict()
        return res
    
    def just_tell(self, clipart, *args, **kwargs):
        assert hasattr(self, 'tell'), "Model is not a teller"
        # if isinstance(self, nn.Module):
        #     self.eval()
        episode = Episode([codraw_data.SelectClipart(clipart)])
        self.tell(episode, *args, **kwargs)
        return episode.get_last(codraw_data.TellGroup).msg
    
    def just_draw(self, msg, scene=[], *args, **kwargs):
        assert hasattr(self, 'draw'), "Model is not a drawer"
        episode = Episode([codraw_data.TellGroup(msg), codraw_data.ObserveCanvas(scene)])
        # if isinstance(self, nn.Module):
        #     self.eval()
        self.draw(episode, *args, **kwargs)
        event_multi = episode.get_last(codraw_data.DrawGroup)
        if event_multi is not None:
            return codraw_data.AbstractScene(event_multi.cliparts)
        
        event_single = episode.get_last(codraw_data.DrawClipart)
        return event_single.clipart


@respond_to(codraw_data.TellGroup)
def drawer_observe_canvas(episode):
    # TODO(nikita): can cache for higher efficiency
    scene = episode.reconstruct()
    event = codraw_data.ObserveCanvas(scene)
    episode.append(event)