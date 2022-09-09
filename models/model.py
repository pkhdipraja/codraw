import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets.codraw_data as codraw_data
import numpy as np

from models.model_utils import Model, drawer_observe_canvas
from datasets.datagen import BOWAddUpdateData
from datasets.episode import respond_to
from datasets.codraw_data import Clipart


class BaseAddOnlyDrawer(Model, nn.Module):
    datagen_cls = BOWAddUpdateData
    def init_full(self, d_hidden):
        # TODO: move to config file?
        # Helps overcome class imbalance (most cliparts are not drawn most of
        # the time)
        self.positive_scaling_coeff = 3.
        # Sigmoid is used to prevent drawing cliparts far off the canvas
        self.sigmoid_coeff = 2.
        # Scaling coefficient so that the sigmoid doesn't always saturate
        self.vals_coeff = 1. / 5.

        dg = self.datagen

        self.canvas_binary_to_hidden = nn.Sequential(
            nn.Dropout(self.cfgs.BINARY_DROPOUT),
            nn.Linear(dg.NUM_BINARY, d_hidden, bias=False),
        )

        self.canvas_numerical_to_hidden = nn.Sequential(
            nn.Linear(dg.NUM_INDEX * dg.NUM_NUMERICAL, d_hidden, bias=False),
            )
        
        d_out = dg.NUM_INDEX * (dg.NUM_ALL + 1)
        self.hidden_to_clipart = nn.Sequential(
            nn.Dropout(self.cfgs.HIDDEN_DROPOUT),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    
    def lang_to_hidden(self, msg_idxs, offsets=None):
        # Offsets is None only when batch_size is 1
        raise NotImplementedError("Subclasses should override this")

    def forward(self, example_batch):
        dg = self.datagen

        hidden_feats = (
            self.lang_to_hidden(example_batch['msg_idxs'], example_batch['offsets'])
            + self.canvas_binary_to_hidden(example_batch['canvas_binary'].float())
            + self.canvas_numerical_to_hidden(example_batch['canvas_numerical'])
            )
        
        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, dg.NUM_ALL + 1)

        correct_categorical = example_batch['clipart_categorical']
        correct_numerical = example_batch['clipart_numerical']
        correct_mask = example_batch['clipart_added_mask']

        clipart_idx_scores = clipart_scores[:,:,0]
        idx_losses = F.binary_cross_entropy_with_logits(clipart_idx_scores, correct_mask.to(torch.float), reduce=False)
        idx_losses = torch.where(correct_mask, self.positive_scaling_coeff * idx_losses, idx_losses)
        per_example_idx_loss = idx_losses.sum(1)

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))

        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * torch.sigmoid(self.vals_coeff * vals_numerical)

        subtype_losses = F.cross_entropy(logits_subtype, correct_categorical[:,:,0].view((-1,)), reduce=False).view_as(correct_categorical[:,:,0])
        depth_losses = F.cross_entropy(logits_depth, correct_categorical[:,:,1].view((-1,)), reduce=False).view_as(correct_categorical[:,:,1])
        flip_losses = F.cross_entropy(logits_flip, correct_categorical[:,:,2].view((-1,)), reduce=False).view_as(correct_categorical[:,:,2])
        vals_losses = F.mse_loss(vals_numerical, correct_numerical.view((-1, dg.NUM_NUMERICAL)), reduce=False).view_as(correct_numerical).sum(-1)
        all_losses = torch.stack([subtype_losses, depth_losses, flip_losses, vals_losses], -1).sum(-1)
        per_example_loss = torch.where(correct_mask, all_losses, all_losses.new_zeros(1)).sum(-1)

        loss = per_example_idx_loss.mean() + per_example_loss.mean()

        return loss
    
    @respond_to(codraw_data.ObserveCanvas)
    def draw(self, episode):
        dg = self.datagen

        msg = episode.get_last(codraw_data.TellGroup).msg
        words = [self.datagen.vocabulary_dict.get(word, None) for word in msg.split()]
        words = [word for word in words if word is not None]
        if not words:
            episode.append(codraw_data.DrawGroup([]))
            episode.append(codraw_data.ReplyGroup("ok"))
            return
        msg_idxs = torch.tensor(words).cuda()

        canvas_context = episode.get_last(codraw_data.ObserveCanvas).scene

        canvas_binary = np.zeros((dg.NUM_INDEX, 1 + dg.NUM_DEPTH + dg.NUM_FLIP), dtype=bool)
        canvas_pose = np.zeros((2, dg.NUM_SUBTYPES), dtype=bool)
        canvas_numerical = np.zeros((dg.NUM_INDEX, dg.NUM_NUMERICAL))

        for clipart in canvas_context:
            if clipart.idx in Clipart.HUMAN_IDXS:
                canvas_pose[clipart.human_idx, clipart.subtype] = True
            
            canvas_binary[clipart.idx, 0] = True
            canvas_binary[clipart.idx, 1 + clipart.depth] = True
            canvas_binary[clipart.idx, 1 + dg.NUM_DEPTH + clipart.flip] = True
            canvas_numerical[clipart.idx, 0] = clipart.normed_x
            canvas_numerical[clipart.idx, 1] = clipart.normed_y
        
        canvas_binary = np.concatenate([canvas_binary.reshape((-1,)), canvas_pose.reshape((-1,))])
        canvas_numerical = canvas_numerical.reshape((-1,))

        canvas_binary = torch.tensor(canvas_binary.astype(np.uint8), dtype=torch.uint8)[None,:].cuda()
        canvas_numerical = torch.tensor(canvas_numerical, dtype=torch.float)[None,:].cuda()

        hidden_feats = (
            self.lang_to_hidden(msg_idxs[None,:], None)
            + self.canvas_binary_to_hidden(canvas_binary.float())
            + self.canvas_numerical_to_hidden(canvas_numerical)
            )
        
        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, (dg.NUM_ALL + 1))

        cliparts = []
        prior_idxs = set([c.idx for c in canvas_context])

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * torch.sigmoid(self.vals_coeff * vals_numerical)
        vals_numerical = vals_numerical.cpu().detach().numpy()

        clipart_idx_scores = clipart_scores[0,:,0].cpu().detach().numpy()

        for idx in np.where(clipart_idx_scores > 0)[0]:
            if idx in prior_idxs:
                continue
            nx, ny = vals_numerical[idx,:]
            clipart = Clipart(idx, int(logits_subtype[idx,:].argmax()), int(logits_depth[idx,:].argmax()), int(logits_flip[idx,:].argmax()), normed_x=nx, normed_y=ny)
            cliparts.append(clipart)
        episode.append(codraw_data.DrawGroup(cliparts))
        episode.append(codraw_data.ReplyGroup("ok"))
    
    def get_action_fns(self):
        return [drawer_observe_canvas, self.draw]
    

class LSTMAddOnlyDrawer(BaseAddOnlyDrawer):
    def init_full(
        self, d_embeddings=None,
        d_hidden=None, d_lstm=None,
        num_lstm_layers=None, pre_lstm_dropout=None,
        lstm_dropout=None
    ):
        self._args = dict(
            d_embeddings=d_embeddings,
            d_hidden=d_hidden,
            d_lstm=d_lstm,
            num_lstm_layers=num_lstm_layers,
            pre_lstm_dropout=pre_lstm_dropout,
            lstm_dropout=lstm_dropout
        )
        super().init_full(d_hidden)

        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.Embedding(len(self.datagen.vocabulary_dict), d_embeddings)
        self.pre_lstm_dropout = nn.Dropout(pre_lstm_dropout)
        self.lstm = nn.LSTM(d_embeddings, d_lstm, bidirectional=True, num_layers=num_lstm_layers, dropout=lstm_dropout)
        # self.post_lstm_project = nn.Linear(d_lstm * 2 * num_lstm_layers, d_hidden)
        # self.post_lstm_project = lambda x: x #nn.Linear(d_lstm * 2 * num_lstm_layers, d_hidden)
        self.post_lstm_project = lambda x: x[:,:d_hidden]
        # self.to(cuda_if_available)
    
    def lang_to_hidden(self, msg_idxs, offsets=None):
        if offsets is not None:
            start = offsets.cpu()
            end = torch.cat([start[1:], torch.tensor([msg_idxs.shape[-1]])])
            undo_sorting = np.zeros(start.shape[0], dtype=int)
            undo_sorting[(start - end).numpy().argsort()] = np.arange(start.shape[0], dtype=int)
            words_packed = nn.utils.rnn.pack_sequence(sorted([msg_idxs[i:j] for i, j in list(zip(start.numpy(), end.numpy()))], key=lambda x: -x.shape[0]))
        else:
            words_packed = nn.utils.rnn.pack_sequence([msg_idxs[0,:]])
            undo_sorting = np.array([0], dtype=int)
        
        word_vecs = embedded = nn.utils.rnn.PackedSequence(
            self.pre_lstm_dropout(self.word_embs(words_packed.data)),
            words_packed.batch_sizes)
        
        _, (h_final, c_final) = self.lstm(word_vecs)
        sentence_reps = c_final[-2:,:,:].permute(1, 2, 0).contiguous().view(undo_sorting.size, -1)
        sentence_reps = self.post_lstm_project(sentence_reps)

        if offsets is not None:
            sentence_reps = sentence_reps[undo_sorting]
        return sentence_reps

