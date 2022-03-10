import torch

def compute_loss(lm_logits, lm_labels, criterion):
    #Language modeling loss
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[:, :-1].contiguous()
    shift_labels = lm_labels[:, 1:].contiguous()

    # Flatten the tokens
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))
    return loss

class LossCompute:
    def __init__(self, lm_criterion,  opt=None,scaler=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt
        self.scaler=scaler

    def __call__(self, lm_logits=None,lm_logits2=None, lm_labels=None,lm_labels2=None, encoder=None, batch_num=None, only_return_losses=False, accum_steps=1):
        if lm_logits is not None:
           loss1 = compute_loss(lm_logits, lm_labels, self.lm_criterion)
        if lm_logits2 is not None:
           loss2 = compute_loss(lm_logits2, lm_labels2, self.lm_criterion)
        if only_return_losses:
            if lm_logits2 is not None:
               return loss1.sum() + loss2.sum()
            else:
               return loss1.sum()
        train_loss = loss1.sum()
        if lm_logits2 is not None:
           train_loss += loss2.sum()
        if self.scaler is not None:
            self.scaler.scale(train_loss).backward()
        else:
            train_loss.backward()
        if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
            print('opt updating')
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class LossComputeSeq2Seq:
    def __init__(self, lm_criterion,  opt=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt

    def __call__(self, train_loss=None,lm_logits2=None, lm_labels=None,lm_labels2=None, encoder=None, batch_num=None, only_return_losses=False, accum_steps=0):
        train_loss = train_loss.sum()
        if not only_return_losses:
            train_loss.backward()
            if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
                print('opt updating')
                self.opt.step()
                self.opt.zero_grad()
        return train_loss.item()