import random
from argparse import Namespace

import random
from argparse import Namespace

import nltk
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_metric
from transformers_lightning.schedulers import LinearSchedulerWithWarmup

try:
    from fairscale.nn import wrap, auto_wrap, default_auto_wrap_policy
except Exception as e:
    print(e)
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, autocast
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, MaxLengthCriteria

from kbs_to_text_unite_summary import additional_tokens, tokens
from model.kbart.modeling_bart_kgc import KBartForConditionalGeneration, ignore_pad_token_for_loss, metric, \
    BartForSequenceClassification
from model.kbart.tokenization_bart import BartTokenizer


class MaxMargin_Loss(torch.nn.Module):
    '''Max margin loss class.  An adaptation of the one utilized in AuxGAN.  '''

    def __init__(self, sim_neg=10, batch_size=32, sim_margin=1):
        super(MaxMargin_Loss, self).__init__()
        # Amount of times to calculate the loss
        self.sim_neg = sim_neg
        self.batch_size = batch_size
        self.sim_margin = sim_margin

    def forward(self, y_pred, y_true):
        cost = 0.
        for i in range(0, self.sim_neg):
            # Gets a random set from the current batch
            new_true = torch.randperm(self.batch_size).to(y_pred.device)
            new_true = y_true[new_true]
            # Normalize everything for a cosine similarity
            normalize_a = self.l2_norm(y_true)
            normalize_b = self.l2_norm(y_pred)
            normalize_c = self.l2_norm(new_true)
            # Cosine similarity, things in the original batch should be close together, things in the other batch
            # should be further apart
            minimize = torch.sum(torch.multiply(normalize_a, normalize_b))
            maximize = torch.sum(torch.multiply(normalize_a, normalize_c))
            # Actual calculation for the loss
            mg = self.sim_margin - minimize + maximize
            # Clamp it at 0 because it can be negative.
            cost += torch.clamp(mg, min=0)
        # Since we are getting the cost for sim_neg, normalize by dividing by the amount
        return cost / self.sim_neg

    def l2_norm(self, x):
        sq = torch.square(x)
        square_sum = torch.sum(torch.sum(sq, dim=1))
        epsilon = 1e-8
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        normalize_a_t = x * x_inv_norm
        return normalize_a_t


class BartGAN(pl.LightningModule):
    # class Discriminator(pl.LightningModule):
    #     def __init__(self, model_path="facebook/bart-base", max_size=385):
    #         super().__init__()
    #         self.encoder = BartModel.from_pretrained(model_path)
    #         self.tokenizer = BartTokenizer.from_pretrained(model_path)
    #         self.expansion_layer = nn.Linear(768,1024)
    #         self.cls_layer = nn.Linear(1024, 1)
    #         self.nonlinear = nn.ReLU()
    #
    #     def forward(self, x):
    #         x_hat = self.encoder(**x)[0]
    #         x_hat = torch.mean(x_hat,1)
    #         x_hat = self.expansion_layer(x_hat)
    #         x_hat = self.nonlinear(x_hat)
    #         x_hat = self.cls_layer(x_hat)
    #         return x_hat  # Y measure without the final classificatiion layer
    class Discriminator(pl.LightningModule):
        def __init__(self, model_path="facebook/bart-base", max_size=385, add_noise=True):
            super().__init__()
            self.encoder = BartForSequenceClassification.from_pretrained(model_path, num_labels=1)
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            # self.reworker = nn.Conv1d(max_size, 128, (2,))
            # self.reworker2 = nn.Conv1d(128, 64, (2,))
            # self.reworker3 = nn.Conv1d(64, 32, (2,))

            # self.maxpool = nn.MaxPool1d(2)
            # self.compressor = nn.Linear(3040, 1024)
            # self.compressor2 = nn.Linear(1024, 512)
            self.cls_layer = nn.Linear(1024, 1)
            self.nonlinear = nn.ReLU()
            self.expansion = nn.Linear(768, 1024)
            self.fusion = nn.Linear(768 + 1024, 1024)
            self.std = 0.01
            self.mean = 0.1
            self.add_noise = add_noise

        def forward(self, x, loss_fn=None, add_noise=False):
            if loss_fn is not None:
                x["loss_fn"] = loss_fn
            if add_noise:
                x["noise"] = True
            x_hat = self.encoder(**x)
            return x_hat  # Y measure without the final classificatiion layer

    class Generator(pl.LightningModule):
        def __init__(self, model_path="facebook/bart-base"):
            super().__init__()
            self.encoder = KBartForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = BartTokenizer.from_pretrained(model_path)

        def forward(self, x, update_mem=None,
                    use_mem=True,
                    clear_mem=True):
            x_hat = self.encoder(**x, update_mem=update_mem,
                                 use_mem=use_mem,
                                 clear_mem=clear_mem)
            return x_hat  # Y measure without the final classificatiion layer

    def __init__(self,
                 generator_base_model_path="facebook/bart-base",
                 discriminator_base_model_path="facebook/bart-base",
                 max_source_length=256,
                 max_target_length=128,
                 total_steps=1,
                 accumulate_batches=1,
                 d_lr=5e-5, g_lr=5e-5, preheat_iters=-1,
                 adversarial_train=True,
                 maxmargingen=True,
                 scheduler=True,
                 train_dis_every_n_iters=5,
                 train_gen_every_n_iters=1,
                 discriminator_train_amount=1,
                 generator_train_amount=1,
                 confounderdis=True,
                 batch_size=32,
                 add_critic_noise=False
                 ):

        super().__init__()
        tokenizer = BartTokenizer.from_pretrained(generator_base_model_path)
        self.automatic_optimization = False
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.hinting = True
        self.adversarial_train = adversarial_train
        self.maxmargingen = maxmargingen
        self.generator = BartGAN.Generator(generator_base_model_path)
        self.generator_tok = tokenizer
        self.discriminator = BartGAN.Discriminator(discriminator_base_model_path, add_noise=False)
        self.scheduler = scheduler
        self.discriminator_tok = self.discriminator.tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # Add the special tokens!
        add_tok = additional_tokens  # + get_rel_tokens(load_dbs(dbs))
        self.discriminator_tok.add_tokens(add_tok)
        self.generator_tok.add_tokens(add_tok)

        self.generator_tok.add_special_tokens({"additional_special_tokens": [tokens[key] for key in tokens.keys()]})
        self.discriminator_tok.add_special_tokens({"additional_special_tokens": [tokens[key] for key in tokens.keys()]})
        self.batch_size = batch_size
        self.generator.encoder.resize_token_embeddings(len(self.generator_tok))
        self.generator.encoder.resize_memory_embeddings(len(self.generator_tok))
        self.discriminator.encoder.resize_token_embeddings(len(self.discriminator_tok))
        self.discriminator_freeze_iters = preheat_iters
        self.scale = nn.Parameter(torch.tensor(1.0, requires_grad=False))
        self.total_steps = total_steps
        self.accumulate_batches = accumulate_batches
        self.test_metric = load_metric("sacrebleu")
        self.test_metric_2 = load_metric('rouge')
        self.test_metric_3 = load_metric('meteor')
        self.ignore_pad_for_loss = True

        self.train_dis_every_n_iters = train_dis_every_n_iters
        self.train_gen_every_n_iters = train_gen_every_n_iters
        self.discriminator_train_amount = discriminator_train_amount
        self.generator_train_amount = generator_train_amount
        self.confounderdis = confounderdis
        self.add_critic_noise = add_critic_noise

    # def configure_sharded_model(self):
    #     try:
    #         # Configure the discriminator
    #         self.discriminator.cls_layer = wrap(self.discriminator.cls_layer)
    #         self.discriminator.expansion = wrap(self.discriminator.expansion)
    #         self.discriminator.fusion = wrap(self.discriminator.fusion)
    #
    #         self.discriminator.encoder.model.encoder.layers = auto_wrap(
    #             self.discriminator.encoder.model.encoder.layers,
    #             auto_wrap_policy=functools.partial(default_auto_wrap_policy, min_num_params=1e6, recurse=True))
    #         self.discriminator.encoder.model.decoder.layers = auto_wrap(
    #             self.discriminator.encoder.model.decoder.layers,
    #             auto_wrap_policy=functools.partial(default_auto_wrap_policy, min_num_params=1e6, recurse=True))
    #
    #     except Exception as e:
    #         print("Could not configure sharded model...", e)

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.generator_tok.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.generator_tok.pad_token_id)
        decoded_labels = self.generator_tok.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.generator_tok.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def forward(self, x):
        embedding = self.generator.encoder(x)
        return embedding

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        if self.scheduler:
            # args = Namespace(num_warmup_steps=int(self.total_steps/ self.train_gen_every_n_iters * 0.1),
            args = Namespace(num_warmup_steps=int(1),
                             max_steps=int(self.total_steps),
                             num_training_steps=int(self.total_steps),
                             scheduler_last_epoch=-1, scheduler_verbose=False)
            lr1 = LinearSchedulerWithWarmup(args, g_opt)

            # args = Namespace(num_warmup_steps=int(self.total_steps * 0.1),
            args = Namespace(num_warmup_steps=int(1),
                             max_steps=self.total_steps,
                             num_training_steps=self.total_steps,
                             scheduler_last_epoch=-1, scheduler_verbose=False)
            lr2 = LinearSchedulerWithWarmup(args, d_opt)

            return [g_opt, d_opt], [lr1, lr2]
        else:
            return [g_opt, d_opt]

    def eval_assertion(self, input_sentences):
        self.eval()
        with torch.no_grad():
            # Prepare for discriminator ingestion
            fake_batch = []
            for i in range(len(input_sentences)):
                # Condition on the input_text and the actual generation
                fake_batch.append(input_sentences[i]["target"])
            fake_cpu = self.discriminator_tok(fake_batch,
                                              max_length=self.max_target_length + self.max_source_length + 1,
                                              padding="max_length",
                                              truncation=True,
                                              return_tensors="pt")
            for i in fake_cpu:
                fake_cpu[i] = fake_cpu[i].to(self.discriminator.device)
            # Add get the encoder representation for the discriminator.
            only_text = self.generator_tok([itm["sources"] for itm in input_sentences],
                                           max_length=self.max_source_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            for i in only_text:
                only_text[i] = only_text[i].to(self.generator.device)

            # Actually evaluate
            output = torch.sigmoid(self.discriminator(fake_cpu)["logits"]).tolist()
            output = [t[0] for t in output]
            return output

    def generate_and_eval(self, input_sentences, update_mem=None, num_beams=10, num_return_sequences=3,
                        max_length=1024):
        self.eval()
        with torch.no_grad():
            # with autocast(device_type=self.device):

                # Prepare for generator
                only_text = self.generator_tok(input_sentences,
                                               max_length=max_length,
                                               padding="max_length",
                                               truncation=True,
                                               return_tensors="pt")
                # Set memory up
                if update_mem is not None:
                    mem_update = []
                    with self.generator_tok.as_target_tokenizer():
                        rework = np.array(update_mem)
                        try:
                            rework = rework.squeeze(1)
                        except:
                            pass
                        rework = rework.transpose()
                        for mem_set in rework:
                            dec_enc_relevant_rels = self.generator_tok(mem_set.tolist(), max_length=128,
                                                                       padding="max_length", truncation=True)

                            mem_update.append(
                                torch.tensor(dec_enc_relevant_rels["input_ids"], device=self.generator.device))
                    mem_update = torch.stack(mem_update, 0)
                    # print(mem_update.shape)
                    mem_update = mem_update.expand((num_beams, mem_update.shape[0], mem_update.shape[1]))
                    update_mem = mem_update
                for key in only_text:
                    try:
                        only_text[key] = only_text[key].to(self.generator.device)
                    except:
                        pass
                # Actually generate
                fake = self.generator.encoder.generate(
                    input_ids=only_text['input_ids'],
                    attention_mask=only_text["attention_mask"],
                    max_length=self.max_target_length,
                    num_beams=num_beams,
                    top_k=120,
                    # top_p=0.9,
                    do_sample=False,
                    update_mem=update_mem,
                    clear_mem=False,
                    num_return_sequences=num_return_sequences
                )
                # Also get the encoder outptus for the discriminator
                # encoder_outputs = torch.mean(self.generator.encoder.model.encoder(only_text['input_ids'], return_dict=True,
                #                                                                   output_hidden_states=True)[
                #                                  "last_hidden_state"], 1)

                # Now clean to send to discriminator for eval.
                fake_texts = []
                for i in range(fake.shape[0]):
                    gids = fake[i, :].tolist()
                    s = self.generator_tok.decode(gids, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=True)  # Replace end of sentence  stuff
                    s = s.replace("<s>", "").replace("</s>", "").replace("</relation>", "").replace("<pad>", "").strip()
                    fake_texts.append(s)
                # Prepare discriminator batch
                fake_batch = []
                for i in range(len(input_sentences)):
                    # Condition on the input_text and the actual generation
                    for j in range(len(fake_texts)):
                        fake_batch.append(input_sentences[i] + self.discriminator.tokenizer.sep_token + fake_texts[j])
                fake_cpu = self.discriminator_tok(fake_batch,
                                                  max_length=self.max_target_length + self.max_source_length + 1,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt")
                for i in fake_cpu:
                    fake_cpu[i] = fake_cpu[i].to(self.discriminator.device)
                # Classify the batch with D to get scores
                output = torch.sigmoid(self.discriminator(fake_cpu)["logits"]).tolist()
                return fake_texts, output

    def training_step(self, train_batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        if self.scheduler:
            lr1, lr2 = self.lr_schedulers()

        # g_sched, d_sched = self.lr_schedulers()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        real_label = torch.ones((len(train_batch["text"]), 1)).to(self.discriminator.device)
        fake_label = torch.zeros((len(train_batch["text"]), 1)).to(self.discriminator.device)
        only_text = self.generator_tok(train_batch["text"],
                                       max_length=self.max_source_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_tensors="pt")
        for i in only_text:
            only_text[i] = only_text[i].to(self.generator.device)

        is_last_batch_to_accumulate = (batch_idx + 1) % self.accumulate_batches == 0 or self.trainer.is_last_batch

        if "mem" in train_batch:
            try:
                update_mem = []
                with self.generator_tok.as_target_tokenizer():
                    rework = np.array(train_batch["mem"])
                    try:
                        rework = rework.squeeze(1)
                    except:
                        pass
                    rework = rework.transpose()
                    for mem_set in rework:
                        mem_set = [x for x in mem_set.tolist() if x!="null:null"]
                        dec_enc_relevant_rels = self.generator_tok(mem_set, max_length=128,
                                                                   padding="max_length", truncation=True)
                        pad = [[self.generator_tok.pad_token_id for padid in range(128) ] for _ in range(45-len(mem_set))]
                        update_mem.append(
                            torch.tensor(dec_enc_relevant_rels["input_ids"]+pad, device=self.generator.device))
                update_mem = torch.stack(update_mem, 0)
            except:
                update_mem = None
        else:
            update_mem = None
        try:
            errD = torch.tensor([0])
            errD_real = torch.tensor([0])
            errD_fake = torch.tensor([0])
            errD_conf = torch.tensor([0])
            d_opt.zero_grad()
            if self.adversarial_train:
                if batch_idx % self.train_dis_every_n_iters == 0:
                    for dis_it in range(self.discriminator_train_amount):
                        real_batch = []
                        for i in range(len(train_batch["text"])):
                            # Condition on the input_text and the actual generation
                            t = train_batch["text"][i] + "<pad>" + train_batch["relation"][i]
                            t = t.replace("</s>","")
                            real_batch.append(t)

                        real_cpu = self.discriminator_tok(real_batch,
                                                          max_length=self.max_target_length + self.max_source_length + 1,
                                                          padding="max_length",
                                                          truncation=True,
                                                          return_tensors="pt")
                        for i in real_cpu:
                            real_cpu[i] = real_cpu[i].to(self.discriminator.device)
                        real_cpu["labels"] = real_label.float()
                        # Forward pass real batch through D
                        output = self.discriminator(real_cpu)
                        # Calculate loss on all-real batch
                        errD_real = output["loss"]
                        # Calculate gradients for D in backward pass
                        self.manual_backward(errD_real)
                        ## Train with all-fake batch
                        for i in only_text:
                            only_text[i] = only_text[i].to(self.generator.device)
                        fake = self.generator.encoder.generate(
                            input_ids=only_text['input_ids'],
                            attention_mask=only_text["attention_mask"],
                            max_length=self.max_target_length,
                            num_beams=3,
                            do_sample=False,
                            top_k=120,
                            top_p=0.9,
                            update_mem=update_mem,
                            use_mem=True,
                            clear_mem=True,
                        )
                        fake_texts = []
                        for i in range(fake.shape[0]):
                            gids = fake[i, :].tolist()
                            s = self.generator_tok.decode(gids, skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=True)  # Replace end of sentence  stuff
                            s = s.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
                            fake_texts.append(s)

                        fake_batch = []
                        for i in range(len(train_batch["text"])):
                            # Condition on the input_text and the actual generation
                            fake_batch.append(train_batch["text"][i] + "<pad>" + fake_texts[i])

                        fb_plus = fake_batch

                        # fb_plus = random.sample(fake_batch, int(len(items) / 2)) + random.sample(falses[len(items):],
                        #                                                                          int(len(items) / 2))
                        fb_plus = [_.replace("</s>","")for _ in fb_plus]

                        fake_cpu = self.discriminator_tok(fb_plus,
                                                          max_length=self.max_target_length + self.max_source_length + 1,
                                                          padding="max_length",
                                                          truncation=True,
                                                          return_tensors="pt")

                        for i in fake_cpu:
                            fake_cpu[i] = fake_cpu[i].to(self.discriminator.device)

                        fake_cpu["labels"] = fake_label.float()

                        output = self.discriminator(fake_cpu)  # Should be detached by here.
                        # Calculate D's loss on the all-fake batch

                        errD_fake = output["loss"]
                        self.manual_backward(errD_fake)

                        if self.confounderdis:
                            # D_G_z1 = output.mean().item()
                            # Compute error of D as sum over the fake and the real batches
                            items = []
                            for i in range(len(train_batch["text"])):
                                # Condition on the input_text and the actual generation
                                t = train_batch["text"][i] + "<pad>" + train_batch["relation"][i]
                                t = t.replace("</s>", "")
                                items.append(t)

                            amount_of_falses = int(len(items))
                            subjects = [item[item.rfind("<subj>") + 6:item.rfind("<obj>")] for item in items]
                            objects = [item[item.rfind("<obj>") + 5:item.rfind("</relation>")] for item in items]
                            random.shuffle(subjects)
                            random.shuffle(objects)
                            falses = []
                            # n, p = 1, .5  # n = coins flipped, p = prob of success

                            for i in range(amount_of_falses):
                                # flip = np.random.binomial(n, p)
                                samp = random.sample(items, 1)[0]
                                # random subject and random object
                                subj = random.sample(subjects, 1)[0]
                                falses.append(samp[:samp.rfind("<subj>") + 6] + subj + samp[samp.rfind("<obj>"):])
                                obj = random.sample(objects, 1)[0]
                                falses.append(samp[:samp.rfind("<obj>") + 5] + obj + samp[samp.rfind("</relation>"):])
                            random.shuffle(items)
                            random.shuffle(falses)
                            # fb_plus = fake_batch

                            fb_plus = random.sample(fake_batch, int(len(items) / 2)) + random.sample(falses, int(len(
                                items) / 2))
                            fb_plus = [_.replace("</s>", "") for _ in fb_plus]

                            fake_cpu = self.discriminator_tok(fb_plus,
                                                              max_length=self.max_target_length + self.max_source_length + 1,
                                                              padding="max_length",
                                                              truncation=True,
                                                              return_tensors="pt")

                            for i in fake_cpu:
                                fake_cpu[i] = fake_cpu[i].to(self.discriminator.device)

                            fake_cpu["labels"] = torch.tensor([[1] for _ in range(int(len(items) / 2))] +
                                                              [[0] for _ in range(int(len(items) / 2))]).to(
                                self.discriminator.device).float()

                            # Classify all fake batch with D
                            output = self.discriminator(fake_cpu)  # Should be detached by here.
                            # Calculate D's loss on the all-fake batch

                            errD_conf = output["loss"]
                            self.manual_backward(errD_conf)

                        else:
                            print("No confounder dis")
                        errD = errD_real + errD_fake + errD_conf.to(errD_fake.device)

                        if is_last_batch_to_accumulate:
                            d_opt.step()
                            lr2.step()

            else:
                if batch_idx % self.train_dis_every_n_iters == 0:
                    for dis_it in range(self.discriminator_train_amount):
                        errD_fake = 0
                        errD_real = 0
                        # D_G_z1 = output.mean().item()
                        # Compute error of D as sum over the fake and the real batches
                        items = []
                        for i in range(len(train_batch["text"])):
                            # Condition on the input_text and the actual generation
                            t = train_batch["text"][i] + "<pad>" + train_batch["relation"][i]
                            t.replace("</s>","")
                            items.append(t)

                        amount_of_falses = int(len(items))
                        # <subj> dust <obj> the refrigerator <\/relation>
                        subjects = [item[item.rfind("<subj>") + 6:item.rfind("<obj>")] for item in items]
                        objects = [item[item.rfind("<obj>") + 5:item.rfind("</relation>")] for item in items]
                        # scp = copy.deepcopy(subjects)
                        # ocp = copy.deepcopy(objects)
                        # while True:
                        random.shuffle(subjects)
                        # if subjects == scp:
                        #     break
                        # while True:
                        random.shuffle(objects)
                        # if objects == ocp:
                        #     break

                        falses = []
                        reals = []
                        # n, p = 1, .5  # n = coins flipped, p = prob of success
                        labels = []
                        for i in range(amount_of_falses):
                            # flip = np.random.binomial(n, p)
                            samp = random.sample(items, 1)[0]
                            reals.append(samp)
                            labels.append([1])
                            # random subject and random object
                            subj = random.sample(subjects, 1)[0]
                            falses.append(samp[:samp.rfind("<subj>") + 6] + subj + samp[samp.rfind("<obj>"):])
                            labels.append([0])
                            obj = random.sample(objects, 1)[0]
                            falses.append(samp[:samp.rfind("<obj>") + 5] + obj + samp[samp.rfind("</relation>"):])
                            labels.append([0])
                        reals_sample = random.sample([i for i in range(len(reals))], int(len(items) / 2))
                        falses_sample = random.sample([i for i in range(len(falses))], int(len(items) / 2))
                        reals = [reals[i] for i in reals_sample]
                        falses = [falses[i] for i in falses_sample]
                        labels = [[1] for _ in range(len(reals))] + [[0] for _ in range(len(falses))]
                        fb_plus = reals + falses
                        fb_plus = [_.replace("</s>","") for _ in fb_plus]
                        fake_cpu = self.discriminator_tok(fb_plus,
                                                          max_length=self.max_target_length + self.max_source_length + 1,
                                                          padding="max_length",
                                                          truncation=True,
                                                          return_tensors="pt")
                        fake_cpu["labels"] = torch.tensor(labels)

                        for i in fake_cpu:
                            fake_cpu[i] = fake_cpu[i].to(self.discriminator.device)
                        fake_cpu["labels"] = fake_cpu["labels"].float()

                        # Classify all fake batch with D
                        output = self.discriminator(fake_cpu)  # Should be detached by here.
                        errD = output["loss"]
                        self.manual_backward(errD)

                        if is_last_batch_to_accumulate:
                            d_opt.step()
                            lr2.step()
                            d_opt.zero_grad()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # Format the targets for loss calc
            g_opt.zero_grad()

            padding = "max_length"
            targets = train_batch["relation"]
            # Setup the tokenizer for targets
            with self.generator_tok.as_target_tokenizer():
                labels = self.generator_tok(targets, max_length=self.max_target_length,
                                            padding=padding, truncation=True)

            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.generator_tok.pad_token_id else -100) for l in label] for label in
                    labels["input_ids"]
                ]

            only_text["labels"] = torch.tensor(labels["input_ids"])
            for i in only_text:
                only_text[i] = only_text[i].to(self.generator.device)

            # Normal LM loss for the gen
            lm_loss = self.generator(only_text,
                                     update_mem=update_mem,
                                     use_mem=True,
                                     clear_mem=True)
            # print("lm loss d")

            errG = lm_loss[0]
            self.manual_backward(lm_loss["loss"])

            debug = False
            # g_opt_conf.zero_grad()
            if debug:
                knn = KNeighborsClassifier(n_neighbors=1)
                vs = self.discriminator.encoder.model.shared.weight.detach().cpu().numpy()
                vsy = [[i] for i in range(vs.shape[0])]
                knn.fit(vs, vsy)

            if (batch_idx + 1) * (self.current_epoch + 1) > self.discriminator_freeze_iters and self.maxmargingen:
                input_ids = only_text["input_ids"]
                attention_mask = self.generator.encoder._prepare_attention_mask_for_generation(
                    input_ids, self.generator_tok.pad_token_id, self.generator_tok.eos_token_id
                )

                model_kwargs = self.generator.encoder.model._prepare_encoder_decoder_kwargs_for_generation(
                    only_text["input_ids"], {})

                input_ids = self.generator.encoder.model._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=self.generator_tok.bos_token_id,
                    bos_token_id=self.generator_tok.bos_token_id
                )
                logits_processor = LogitsProcessorList([
                    MinLengthLogitsProcessor(5, eos_token_id=self.generator.encoder.config.eos_token_id),
                ])

                fake = self.generator.encoder.greedy_search(
                    input_ids,
                    attention_mask=attention_mask,
                    stopping_criteria=MaxLengthCriteria(self.max_target_length),
                    logits_processor=logits_processor,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                    eos_token_id=self.generator.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **model_kwargs
                )
                a = fake["scores"]
                vectors = []
                for ix, s in enumerate(a):
                    # self.scale = self.scale.to(s.device)

                    sms = nn.functional.softmax(torch.clamp(self.scale, 1, 200) * s, dim=1)
                    tst = torch.tensordot(sms, self.discriminator.encoder.model.shared.weight, dims=1)
                    vectors.append(tst)
                fake_vectors = torch.stack(vectors, 1)
                # for i in range(fake_vectors.shape[0]):
                #     predsknn = knn.predict(fake_vectors[i,:,:].detach().cpu().numpy())
                #     decpreds = self.discriminator_tok.decode(list(predsknn))

                seqs = fake["sequences"].tolist()
                pad_idxs = []
                for seq in seqs:
                    try:
                        pad_idxs.append(seq.index(self.generator_tok.pad_token_id) - 1)
                    except:
                        pad_idxs.append(len(seq) - 1)
                # for bi in range(fake_vectors.shape[0]):
                #     dm = fake_vectors[bi,0:pad_idxs[bi],:]
                fake_vectors_avg_til_pad = [torch.mean(fake_vectors[bi, 0:pad_idxs[bi], :], 0).clone() for bi in
                                            range(fake_vectors.shape[0])]
                dum_labels = only_text["labels"]
                dum_labels[dum_labels == -100] = self.generator_tok.pad_token_id
                pad_idxs = []
                for seq in dum_labels.tolist():
                    try:
                        pad_idxs.append(seq.index(self.generator_tok.pad_token_id) - 1)
                    except:
                        pad_idxs.append(len(seq) - 1)

                gol_vecs = self.generator.encoder.model.shared(dum_labels)
                gol_vectors_avg_til_pad = [torch.mean(gol_vecs[bi, 0:pad_idxs[bi], :], 0).clone() for bi in
                                           range(gol_vecs.shape[0])]
                mm_loss = MaxMargin_Loss(batch_size=gol_vecs.shape[0])(torch.stack(fake_vectors_avg_til_pad),
                                                                       torch.stack(gol_vectors_avg_til_pad))
                errD_conf = mm_loss
                errG += errD_conf
                # self.manual_backward(mm_loss, retain_graph=False)

            if (batch_idx + 1) * (
                    self.current_epoch + 1) > self.discriminator_freeze_iters and self.adversarial_train and batch_idx % self.train_gen_every_n_iters == 0:

                # g_opt_adv.zero_grad()

                input_ids = only_text["input_ids"]
                attention_mask = self.generator.encoder._prepare_attention_mask_for_generation(
                    input_ids, self.generator_tok.pad_token_id, self.generator_tok.eos_token_id
                )

                model_kwargs = self.generator.encoder.model._prepare_encoder_decoder_kwargs_for_generation(
                    only_text["input_ids"], {})

                input_ids = self.generator.encoder.model._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=self.generator_tok.bos_token_id,
                    bos_token_id=self.generator_tok.bos_token_id
                )
                logits_processor = LogitsProcessorList([
                    MinLengthLogitsProcessor(5, eos_token_id=self.generator.encoder.config.eos_token_id),
                ])

                fake = self.generator.encoder.greedy_search(
                    input_ids,
                    attention_mask=attention_mask,
                    stopping_criteria=MaxLengthCriteria(self.max_target_length),
                    logits_processor=logits_processor,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                    eos_token_id=self.generator.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **model_kwargs
                )
                a = fake["scores"]
                vectors = []
                for ix, s in enumerate(a):
                    # self.scale = self.scale.to(s.device)
                    sms = nn.functional.softmax(torch.clamp(self.scale, 1, 200) * s, dim=1)
                    tst = torch.tensordot(sms, self.discriminator.encoder.model.shared.weight, dims=1)
                    vectors.append(tst)
                fake_vectors = torch.stack(vectors, 1)
                # We have the 127 vectors that correspond to the generated output,

                fake_vectors = fake_vectors.to(
                    self.discriminator.device)  # We have the 127 vectors that correspond to the generated output,
                self.discriminator.encoder.model.shared.weight.to(self.discriminator.device)
                # now we need to join that with the vectors that correspond to the conditioned text and the separator
                bos_vec = self.discriminator.encoder.model.shared(
                    torch.tensor([self.discriminator_tok.bos_token_id]).to(
                        self.discriminator.device)) * self.discriminator.encoder.model.encoder.embed_scale
                eos_vec = self.discriminator.encoder.model.shared(
                    torch.tensor([self.discriminator_tok.eos_token_id]).to(
                        self.discriminator.device)) * self.discriminator.encoder.model.encoder.embed_scale
                sep_vec = self.discriminator.encoder.model.shared(
                    torch.tensor([self.discriminator_tok.encode("</relation>")[0]]).to(
                        self.discriminator.device)) * self.discriminator.encoder.model.encoder.embed_scale
                pad_vec = self.discriminator.encoder.model.shared(
                    torch.tensor([self.discriminator_tok.pad_token_id]).to(
                        self.discriminator.device)) * self.discriminator.encoder.model.encoder.embed_scale
                # if batch_idx%1000==0:
                fake_texts = []
                real_stuff = []
                for i in range(fake_vectors.shape[0]):
                    # print(fake_vectors[i,:,:].shape,i,fake_vectors.shape[0])
                    gids = fake["sequences"][i, 1:].tolist()
                    translation = self.discriminator_tok.decode(gids)
                    to_remove = [j for j in range(0, len(gids)) if gids[j] == self.discriminator_tok.pad_token_id] + \
                                [j for j in range(0, len(gids)) if gids[j] == self.discriminator_tok.bos_token_id] + \
                                [j for j in range(0, len(gids)) if gids[j] == self.discriminator_tok.eos_token_id] + \
                                [j for j in range(0, len(gids)) if gids[j] == self.discriminator_tok.sep_token_id]
                    to_remove = sorted(list(set(to_remove)))
                    to_keep = [j for j in range(0, len(gids)) if j not in to_remove]
                    if debug:
                        tst = [gids[itm] for itm in to_keep]
                        translation2 = self.discriminator_tok.decode(tst)
                        dmsmd = fake_vectors[i, to_keep, :]
                        dmsmd2 = self.discriminator.encoder.model.shared(
                            torch.tensor([[tsts] for tsts in [gids[itm] for itm in to_keep]]).to(
                                self.discriminator.encoder.model.shared.weight.device))
                        predsknn = knn.predict(dmsmd.detach().cpu().numpy())
                        decpreds = self.discriminator_tok.decode(list(predsknn))
                    # predsknn2 = knn.predict(dmsmd2.detach().cpu().numpy())
                    # decpreds2 = self.discriminator_tok.decode(list(predsknn2))
                    fake_texts.append(fake_vectors[i, to_keep, :])
                fake_batch = []
                for i in range(len(train_batch["text"])):
                    # Condition on the input_text and the actual generation
                    fake_batch.append(self.discriminator.encoder.model.shared(
                        self.discriminator_tok(train_batch["text"][i],
                                               add_special_tokens=False,
                                               truncation=True,
                                               max_length=self.max_source_length,
                                               return_tensors="pt")["input_ids"].to(self.discriminator.device)
                    ) * self.discriminator.encoder.model.encoder.embed_scale)

                final_fake_batch = {
                    "inputs_embeds": [],
                    "attention_mask": [],
                    "eos_mask": []
                }

                for i in range(len(train_batch["text"])):
                    vecs = [bos_vec, fake_batch[i].squeeze(0), pad_vec, fake_texts[i]]
                    vecs = torch.cat(vecs, 0)
                    vecs = vecs[0:self.max_target_length + self.max_source_length, :]
                    vecs = torch.cat([vecs, eos_vec], 0)
                    eos_idx = vecs.shape[0] - 1

                    tot = vecs.shape[0]
                    att = [1 for i in range(0, tot)] + [0 for i in
                                                        range(0,
                                                              self.max_target_length + self.max_source_length + 1 - tot)]
                    pvecs = [pad_vec.clone().detach() for i in
                             range(0, self.max_target_length + self.max_source_length + 1 - tot)]
                    for pvec in pvecs:
                        pvec.requires_grad = False
                    vecs = torch.cat(
                        [vecs] + pvecs, 0)
                    att = torch.tensor(att).to(self.discriminator.device)
                    #     # print("Current predictions fed as GAN training")
                    #
                    if debug:
                        dmsmd = vecs
                        predsknn = knn.predict(dmsmd.detach().cpu().numpy())
                        decpreds = self.generator_tok.decode(list(predsknn))

                    final_fake_batch["inputs_embeds"].append(vecs)

                    final_fake_batch["attention_mask"].append(torch.tensor(att))
                    eos = torch.zeros(len(vecs))

                    eos[eos_idx] = 1

                    final_fake_batch["eos_mask"].append(eos)

                for i in final_fake_batch:
                    final_fake_batch[i] = torch.stack(final_fake_batch[i], 0)
                    final_fake_batch[i] = final_fake_batch[i].to(self.discriminator.device)
                if self.add_critic_noise:
                    final_fake_batch["inputs_embeds"]+=torch.randn(final_fake_batch["inputs_embeds"].shape).to(final_fake_batch["inputs_embeds"].device)*0.01
                final_fake_batch["labels"] = real_label
                output = self.discriminator(final_fake_batch)
                # print(output)
                critic_loss = output["loss"]
                # critic_loss = F.binary_cross_entropy_with_logits(output, real_label)
                self.manual_backward(critic_loss)

                errG += critic_loss

            else:
                critic_loss = torch.tensor([0]).to(lm_loss[0].device)
                errG = lm_loss[0]
                # self.manual_backward(errG)

            if is_last_batch_to_accumulate:
                lr1.step()
                g_opt.step()

            loss_dict = {
                "loss": errG + errD.to(errG.device),
                "g_loss": errG,
                "g_critic_loss": critic_loss,
                "d_loss": errD,
                "errD_real": errD_real,
                "errD_fake": errD_fake,
                "errD_conf": errD_conf,
                "lm_loss": lm_loss[0],
                "batch_size": self.batch_size
            }
        except Exception as e:
            print("ERROR", e)
            loss_dict = {
                "loss": 0,
                "g_loss": 0,
                "g_critic_loss": 0,
                "d_loss": 0,
                "errD_real": 0,
                "errD_fake": 0,
                "errD_conf": 0,
                "lm_loss": 0,
                "batch_size": self.batch_size
            }
        for key in loss_dict:
            try:
                conv = torch.tensor(loss_dict[key])
                loss_dict[key] = conv
            except:
                continue
        if is_last_batch_to_accumulate:
            self.log_dict(loss_dict, prog_bar=True)

        return loss_dict


    def test_step(self, test_batch, test_batch_idx):
        only_text = self.generator_tok(test_batch["text"],
                                       max_length=self.max_source_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_tensors="pt")
        padding = "max_length"
        # Setup the tokenizer for targets
        with self.generator_tok.as_target_tokenizer():
            labels = self.generator_tok(test_batch["relation"], max_length=self.max_target_length,
                                        padding=padding, truncation=True)

        if padding == "max_length" and self.ignore_pad_for_loss:
            labels["input_ids"] = [
                (label if label != self.generator_tok.pad_token_id else -100) for label in labels["input_ids"]
            ]

        only_text["labels"] = torch.tensor(labels["input_ids"])
        update_mem = None
        for i in only_text:
            only_text[i] = only_text[i].to(self.generator.device)

        gen = self.generator.encoder.generate(
            input_ids=only_text['input_ids'],
            attention_mask=only_text["attention_mask"],
            max_length=self.max_target_length,
            num_beams=1,
            do_sample=False,
            update_mem=update_mem,
            use_mem=False
        )
        bl = only_text["labels"].tolist()
        # Fix for pad token not being accounted in loss
        fix_bl = []
        for b in bl:
            fix_bl.append([bli if bli != -100 else self.generator_tok.pad_token_id for bli in b])

        decodings = self.generator_tok.batch_decode(fix_bl, skip_special_tokens=False)
        decodings = [[d.replace("<s>", "").replace("</s>", "").replace("<pad>", "")] for d in decodings]

        gen_rel = self.generator_tok.batch_decode(gen.tolist(), skip_special_tokens=False)
        gen_rel = [d.replace("<s>", "").replace("</s>", "").replace("<pad>", "") for d in gen_rel]
        if test_batch_idx % 10 == 0:
            print("Generations", gen_rel)
            print("Gold", decodings)
        self.test_metric.add_batch(predictions=gen_rel, references=decodings)
        for g, d in zip(gen_rel, [decodings[i][0] for i in range(len(decodings))]):
            self.test_metric_2.add_batch(predictions=[g], references=[d])
            self.test_metric_3.add_batch(predictions=[g], references=[d])


    def on_test_epoch_end(self):
        s = self.test_metric.compute()["score"]
        r = self.test_metric_2.compute(use_stemmer=True)
        rouge_result = {key: value.mid.fmeasure * 100 for key, value in r.items()}
        m = self.test_metric_3.compute()['meteor'] * 100
        results = {
            "meteor": m,
            "bleu": s
        }
        results.update(rouge_result)
        print("test_bleu", s)
        print("test_rouge", r)
        print("test_meteor", m)
        self.log_dict(results, sync_dist=True)
