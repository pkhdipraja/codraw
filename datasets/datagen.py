import torch
import numpy as np
import datasets.codraw_data as codraw_data

from torch.utils.data import Dataset
from datasets.codraw_data import Clipart


class Datagen(Dataset):
    # the spec contains summaries (like a vocab list), but the events are stored
    # as a pointer and not as the actual events dictionary. The events get
    # restored only if needed, (which shouldn't really be the case because saved
    # models won't need to be trained further.)
    def __init__(self, cfgs, split=None, spec=None, **kwargs):
        super(Datagen, self).__init__()
        # self._examples_cache = None
        self.cfgs = cfgs
        if spec is not None:
            self.split = spec['split']
            self.init_from_spec(**{k: v for (k, v) in spec.items() if k != 'split'})
        else:
            self.split = split
            self.init_full(**kwargs)

        self.examples = list(self.get_examples())

    def init_full(self):
        raise NotImplementedError("Subclasses should override this")

    def init_from_spec(self):
        raise NotImplementedError("Subclasses should override this")

    def __len__(self):
        raise NotImplementedError("Subclasses should override this")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should override this")

    def calc_derived(self):
        pass

    def get_spec(self):
        return {}

    @property
    def spec(self):
        spec = self.get_spec()
        if 'split' not in spec:
            spec['split'] = self.split
        return spec

    def get_examples(self):
        raise NotImplementedError("Subclasses should override this")
    
    def vocabulary_for_split(self, split, event_getter=codraw_data.get_place_one):
        vocabulary = set()

        it = iter(event_getter(self.cfgs, split))
        for event in it:
            if isinstance(event, codraw_data.TellGroup):
                msg = event.msg
                vocabulary |= set(msg.split())

        return sorted(vocabulary)

    # def collate(self, batch):
    #     raise NotImplementedError("Subclasses should override this")

    # def get_examples_batch(self, batch_size=16):
    #     if self._examples_cache is None:
    #         self._examples_cache = list(self.get_examples())

    #     batch = []
    #     epoch_examples = self._examples_cache[:]
    #     np.random.shuffle(epoch_examples)
    #     for ex in epoch_examples:
    #         batch.append(ex)
    #         if len(batch) == batch_size:
    #             yield self.collate(batch)
    #             batch = []

    # def get_examples_unshuffled_batch(self, batch_size=16):
    #     """
    #     Does not shuffle, and the last batch may contain less elements.
    #     Originally added for perplexity evaluation.
    #     """
    #     if self._examples_cache is None:
    #         self._examples_cache = list(self.get_examples())

    #     batch = []
    #     epoch_examples = self._examples_cache[:]
    #     for ex in epoch_examples:
    #         batch.append(ex)
    #         if len(batch) == batch_size:
    #             yield self.collate(batch)
    #             batch = []

    #     if batch:
    #         yield self.collate(batch)


class BOWAddUpdateData(Datagen):
    NUM_INDEX = Clipart.NUM_IDX
    NUM_SUBTYPES = Clipart.NUM_SUBTYPE
    NUM_DEPTH = Clipart.NUM_DEPTH
    NUM_FLIP = Clipart.NUM_FLIP
    NUM_CATEGORICAL = NUM_SUBTYPES + NUM_DEPTH + NUM_FLIP
    NUM_NUMERICAL = 2  # x, y

    NUM_ALL = NUM_CATEGORICAL + NUM_NUMERICAL

    NUM_BINARY = (NUM_INDEX * (1 + NUM_DEPTH + NUM_FLIP)) + 2 * NUM_SUBTYPES

    NUM_X_TICKS = 3
    NUM_Y_TICKS = 2
    NUM_TAGS = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + NUM_X_TICKS + NUM_Y_TICKS + 1
    NUM_TAGS_PER_INDEX = 6  # index, subtype, depth, flip, x, y

    def init_full(self):
        self.vocabulary = self.vocabulary_for_split(self.split, codraw_data.get_contextual_place_many)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def get_examples(self):
        it = iter(codraw_data.get_contextual_place_many(self.cfgs, self.split))
        for event in it:
            if isinstance(event, codraw_data.TellGroup):
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.ObserveCanvas)
                canvas_context = event.scene
                event = next(it)
                assert isinstance(event, codraw_data.DrawGroup)
                cliparts = event.cliparts
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                if not msg:
                    continue

                context_idxs = set([c.idx for c in canvas_context])

                clipart_added_mask = np.zeros(self.NUM_INDEX , dtype=bool)
                clipart_updated_mask = np.zeros(self.NUM_INDEX , dtype=bool)
                clipart_categorical = np.zeros((self.NUM_INDEX, 3))
                clipart_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))

                for clipart in cliparts:
                    if clipart.idx in context_idxs:
                        clipart_updated_mask[clipart.idx] = True
                    else:
                        clipart_added_mask[clipart.idx] = True
                    clipart_categorical[clipart.idx, :] = [clipart.subtype, clipart.depth, clipart.flip]
                    clipart_numerical[clipart.idx, :] = [clipart.normed_x, clipart.normed_y]

                # TODO: maybe can be combined with the step above?
                clipart_added_mask = torch.tensor(clipart_added_mask.astype(np.uint8), dtype=torch.uint8)
                clipart_updated_mask = torch.tensor(clipart_updated_mask.astype(np.uint8), dtype=torch.uint8)
                clipart_categorical = torch.tensor(clipart_categorical, dtype=torch.long)
                clipart_numerical = torch.tensor(clipart_numerical, dtype=torch.float)

                canvas_binary = np.zeros((self.NUM_INDEX, 1 + self.NUM_DEPTH + self.NUM_FLIP), dtype=bool)
                canvas_pose = np.zeros((2, self.NUM_SUBTYPES), dtype=bool)
                canvas_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))
                canvas_tags = np.zeros((self.NUM_INDEX + 1, self.NUM_TAGS_PER_INDEX), dtype=int)
                canvas_mask = np.zeros(self.NUM_INDEX + 1, dtype=bool)

                for clipart in canvas_context:
                    if clipart.idx in Clipart.HUMAN_IDXS:
                        canvas_pose[clipart.human_idx, clipart.subtype] = True

                    canvas_binary[clipart.idx, 0] = True
                    canvas_binary[clipart.idx, 1 + clipart.depth] = True
                    canvas_binary[clipart.idx, 1 + self.NUM_DEPTH + clipart.flip] = True
                    canvas_numerical[clipart.idx, 0] = clipart.normed_x
                    canvas_numerical[clipart.idx, 1] = clipart.normed_y

                    x_tick = int(np.floor(clipart.normed_x * self.NUM_X_TICKS))
                    if x_tick < 0:
                        x_tick = 0
                    elif x_tick >= self.NUM_X_TICKS:
                        x_tick = self.NUM_X_TICKS - 1

                    y_tick = int(np.floor(clipart.normed_y * self.NUM_Y_TICKS))
                    if y_tick < 0:
                        y_tick = 0
                    elif y_tick >= self.NUM_Y_TICKS:
                        y_tick = self.NUM_Y_TICKS - 1

                    # Tag features (for attention)
                    canvas_tags[clipart.idx, 0] = 1 + clipart.idx
                    canvas_tags[clipart.idx, 1] = 1 + Clipart.NUM_IDX + clipart.subtype
                    canvas_tags[clipart.idx, 2] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + clipart.depth
                    canvas_tags[clipart.idx, 3] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + int(clipart.flip)
                    canvas_tags[clipart.idx, 4] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + x_tick
                    canvas_tags[clipart.idx, 5] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + self.NUM_X_TICKS + y_tick

                    canvas_mask[clipart.idx] = True

                if not canvas_context:
                    canvas_tags[-1, 0] = self.NUM_TAGS - 1
                    canvas_mask[-1] = True

                canvas_binary = np.concatenate([canvas_binary.reshape((-1,)), canvas_pose.reshape((-1,))])
                canvas_numerical = canvas_numerical.reshape((-1,))

                canvas_binary = torch.tensor(canvas_binary.astype(np.uint8), dtype=torch.uint8)
                canvas_numerical = torch.tensor(canvas_numerical, dtype=torch.float)

                canvas_tags = torch.tensor(canvas_tags, dtype=torch.long)
                canvas_mask = torch.tensor(canvas_mask.astype(np.uint8), dtype=torch.uint8)

                msg_idxs = [self.vocabulary_dict.get(word, None) for word in msg.split()]
                msg_idxs = [idx for idx in msg_idxs if idx is not None]

                msg_idxs = torch.LongTensor(msg_idxs)
                example = {
                    'clipart_added_mask': clipart_added_mask,
                    'clipart_updated_mask': clipart_updated_mask,
                    'clipart_categorical': clipart_categorical,
                    'clipart_numerical': clipart_numerical,
                    'canvas_binary': canvas_binary,
                    'canvas_numerical': canvas_numerical,
                    'canvas_tags': canvas_tags,
                    'canvas_mask': canvas_mask,
                    'msg_idxs': msg_idxs,
                }

                yield example



    def __len__(self):
        return self.examples.__len__()

    def __getitem__(self, idx):
        return self.examples[idx]


# def vocabulary_for_split(split, event_getter=codraw_data.get_place_one):
#     vocabulary = set()

#     it = iter(event_getter(split))
#     for event in it:
#         if isinstance(event, codraw_data.TellGroup):
#             msg = event.msg
#             vocabulary |= set(msg.split())

#     return sorted(vocabulary)


def custom_collate(batch):
    # default collate function already handles dict, but we also need to
    # include offset here
    offsets = np.cumsum([0] + [len(x['msg_idxs']) for x in batch])[:-1]

    return {
        'clipart_added_mask': torch.stack([x['clipart_added_mask'] for x in batch]),
        'clipart_updated_mask': torch.stack([x['clipart_updated_mask'] for x in batch]),
        'clipart_categorical': torch.stack([x['clipart_categorical'] for x in batch]),
        'clipart_numerical': torch.stack([x['clipart_numerical'] for x in batch]),
        'canvas_binary': torch.stack([x['canvas_binary'] for x in batch]),
        'canvas_numerical': torch.stack([x['canvas_numerical'] for x in batch]),
        'canvas_tags': torch.stack([x['canvas_tags'] for x in batch]),
        'canvas_mask': torch.stack([x['canvas_mask'] for x in batch]),
        'msg_idxs': torch.cat([x['msg_idxs'] for x in batch]),
        'offsets': torch.tensor(offsets)
    }
