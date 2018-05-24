import numpy as np
import torch
from torch.autograd import Variable
import const
np.random.seed(5)
class DataLoader(object):
    def __init__(self, src_sents, label,mask,mask_all, max_len, cuda=True,
                batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        self._mask = np.asarray(mask)
        self._mask_all = np.asarray(mask_all)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.seed(5)
        np.random.shuffle(indices)
        #print self._src_sents
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]
        self._mask = self._mask[indices]
        self._mask_all = self._mask_all[indices]

    def __iter__(self):
        return self

    def next(self):
        self._shuffle()
        '''
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        #print '_shuffle'
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]
        '''
        #print self._mask,self._label
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=self.evaluation)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        #print _start,_bsz
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start+_bsz], self._max_len)
        label = Variable(torch.from_numpy(self._label[_start:_start+_bsz]),
                    volatile=self.evaluation)
        #print self._src_sents[:3],data,label,self._mask[:3]
        mask=Variable(torch.from_numpy(self._mask[_start:_start+_bsz]),
                    volatile=self.evaluation)
        mask_all=Variable(torch.from_numpy(self._mask_all[_start:_start+_bsz]),
                    volatile=self.evaluation)
        if self.cuda:
            label = label.cuda()
            mask = mask.cuda()
            mask_all = mask_all.cuda()
        return data, label,mask,mask_all