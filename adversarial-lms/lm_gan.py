print("Starting imports")
import argparse
import json
import random

print("Importing numpy")
import numpy as np
print("Import torch")
import torch

print("Importing pl")
import pytorch_lightning as pl
print("seed")
from pytorch_lightning import seed_everything
print("mc, lr")
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
print("wandb")
from pytorch_lightning.loggers import WandbLogger
print("Import dl, ds")
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)
# from model.kbart.kgcbartgan import BartGAN
from model.kbart.kgcbartgannormalgan import BartGAN
print("Done imports")

class ContextualizedRelations(Dataset):
    def __init__(self, file='train_united_seq2seq.json', use_mem=False, mem_file="additional_facts.json",
                 fact_list="fact_list.json", is_test=False, limit=None, random_hints=True, test=False,
                 hint_subject=False, hint_object=False, hint_relation=False, hint_specificity=False):
        self.data = []
        self.is_test = is_test
        self.use_mem = False if self.is_test else use_mem
        self.fact_list = None
        self.test = is_test
        self.random_hints = random_hints

        self.hint_subject = hint_subject
        self.hint_object = hint_object
        self.hint_relation = hint_relation
        self.hint_specificity = hint_specificity

        self.additional_fact_map = None
        if self.use_mem:
            print("Loading fact list")
            self.fact_list = json.load(open(fact_list))
            print('LOADING map')
            self.additional_fact_map = json.load(open(mem_file))
            print("cleaning")
            for key in tqdm(self.additional_fact_map):
                self.additional_fact_map[key] = [x[0] for x in self.additional_fact_map[key]]
            print("Amount of similar facts", len(self.fact_list))
            test_map = dict(zip(self.fact_list, self.additional_fact_map.values()))

        count = 0
        skipped = 0
        with open(file) as f:
            for line in tqdm(f):
                if limit is not None:
                    if count == limit:
                        break
                ld = json.loads(line)
                if use_mem and not is_test:
                    try:
                        t = test_map[ld["relation"]]
                    except:
                        skipped += 1
                        continue

                if is_test:
                    if not ("(" in ld["text"] and ")" in ld["text"]):
                        self.data.append(ld)
                else:
                    for hint in ["<subj>", "<relation>", "<obj>", "<general>", "<specific>"]:
                        if hint in ld["text"]:
                            skipped += 1
                            continue
                    self.data.append(ld)
                count += 1
        print("Skipped", skipped)

        for d in self.data:
            if d["relation"] == "nan" or " nan " in d["relation"]:
                print("FUCK")
        print("Amount of lines in ", file, len(self.data))
        self.encoded_facts = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i = self.data[idx]
        i["relation"] = str(i["relation"])
        i["text"] = str(i["text"])
        targets = i["relation"]
        if not self.is_test:
            if self.random_hints or self.hint_subject or self.hint_relation or self.hint_object:  # Load up all the hint components
                hint_components = i["relation"].split("<")[1:-1]
                hint_components = ["<" + itm for itm in hint_components]
                hst = [self.hint_specificity, self.hint_relation, self.hint_subject, self.hint_object]
                hint_components = [hint_components[hidx] for hidx, hint_part in enumerate(hst) if hint_part]

            if self.random_hints:  # Randomly select these components
                random_hint = np.random.binomial(1, 0.5, 1)[0]
                amount_to_hint = random.sample([1, 2, 3], 1)[0]
                random_hint_components = random.sample([t for t in range(len(hint_components))], amount_to_hint)
                random_hint_components = sorted(random_hint_components)
                random_hint_components = [hint_components[t] for t in random_hint_components]
                i["text"] += (" ( " + ';'.join(random_hint_components) + " ) " if random_hint == 1 else "")
            elif self.hint_subject or self.hint_relation or self.hint_object or self.hint_specificity:  # no randomization always supply those components
                hst = [self.hint_specificity, self.hint_relation, self.hint_subject, self.hint_object]
                tst = [hint_components[hidx] for hidx, hint_part in enumerate(hst) if hint_part]
                i["text"] += " ( " + ';'.join(tst) + ";) "
        else:
            hint_components = i["relation"].split("<")[1:-1]
            hint_components = ["<" + itm for itm in hint_components]
            hst = [self.hint_specificity, self.hint_relation, self.hint_subject, self.hint_object]
            hint_components = [hint_components[hidx] for hidx, hint_part in enumerate(hst) if hint_part]
            if len(hint_components) > 0:
                i["text"] += " ( " + ';'.join(hint_components) + " ) "
        if self.use_mem:
            max_mem = 45
            i["mem"] = [self.fact_list[x] for x in self.additional_fact_map[str(self.fact_list.index(i["relation"]))]][0:max_mem]
            i["mem"]+=["null:null" for _ in range(max_mem-len(i["mem"]))]
            # print(len(i["mem"]))
        return i


if __name__ == '__main__':
    print("Starting main...")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='amount of epochs')
    parser.add_argument('--log_every_n_steps', type=int, default=5,
                        help='amount of steps before logging')
    parser.add_argument('--d_lr', type=float, default=1e-5,
                        help='discriminator learning rate')
    parser.add_argument('--g_lr', type=float, default=1e-5,
                        help='generator learning rate')
    parser.add_argument('--preheat_iters', type=int, default=-1,
                        help='iterations to warmup the generator, -1 is none')

    parser.add_argument('--num_processes', type=int, default=0,
                        help='amount of processes for dataloader')
    parser.add_argument('--accumulate_batches', type=int, default=1,
                        help='amount of batches to accumulate before backprop')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for experiment')
    parser.add_argument('--limit', default=None, type=int,
                        help='limit amount for the loaded data')
    parser.add_argument('output_folder',
                        help='Path to where to save!')

    parser.add_argument('--resumepath', default=None,
                        help='Path to a chekpoint to resume!')
    parser.add_argument('--plugins', default=None,
                        help='plugins to use')
    parser.add_argument('--accelerator', default=None,
                        help='accelerator to use')

    parser.add_argument('--use_mem', action="store_true", default=False,
                        help='whether to use memory or not')
    parser.add_argument('--do_train', action="store_true", default=False,
                        help='whether to train or not')
    parser.add_argument('--do_test', action="store_true", default=False,
                        help='whether to test or not')

    parser.add_argument('--adversarial', action="store_true", default=False,
                        help='whether to train adversarially')
    parser.add_argument('--maxmargingen', action="store_true", default=False,
                        help='Use max margin loss in generation')
    parser.add_argument('--confounderdis', action="store_true", default=False,
                        help='Use confounder loss in discriminator')

    parser.add_argument('--scheduler', action="store_true", default=False,
                        help='Use a linear scheduler with 10% warmup.')
    parser.add_argument('--add_critic_noise', action="store_true", default=False,
                        help='Add noise to the critic input on the generator.')
    parser.add_argument('--train_dis_every_n_iters', default=1, type=int,
                        help='amount of iterations that happen between a training batch for a disciminator')
    parser.add_argument('--train_gen_every_n_iters', default=5, type=int,
                        help='amount of iterations that happen between a training batch for a generator')
    parser.add_argument('--discriminator_train_amount', default=1, type=int,
                        help='amount of times to train the disciminator per train batch')
    parser.add_argument('--generator_train_amount', default=1, type=int,
                        help='amount of times to train the disciminator per train batch')

    parser.add_argument('--fp16', action="store_true", default=False,
                        help='whether to use fp16 or not.')

    parser.add_argument('--train_file', default="train_united_seq2seq.json",
                        help='Path to a chekpoint to resume!')
    parser.add_argument('--test_file', default="test_united_seq2seq.json",
                        help='Path to a chekpoint to resume!')


    args = parser.parse_args()
    if args.seed is not None:
        seed = args.seed
        seed_everything(seed)
    batch_size = args.batch_size
    training_data = ContextualizedRelations(file=args.train_file,use_mem=args.use_mem, mem_file="additional_facts.json",
                                            fact_list="fact_list.json", limit=args.limit, hint_specificity=True,
                                            hint_subject=True, hint_object=True, random_hints=True, hint_relation=True)
    eval_data = ContextualizedRelations(file=args.test_file,
                                        use_mem=args.use_mem,
                                        mem_file="additional_facts.json", fact_list="fact_list.json",
                                        is_test=True)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(eval_data, batch_size=batch_size)
    # model
    print("Creating the model")
    total_steps = int((len(train_loader) * args.epochs) / args.accumulate_batches / (
        1))
    print("Total steps ", total_steps)
    model = BartGAN(total_steps=total_steps,
                    accumulate_batches=args.accumulate_batches,
                    d_lr=args.d_lr, g_lr=args.g_lr,
                    preheat_iters=args.preheat_iters,
                    adversarial_train=args.adversarial,
                    maxmargingen=args.maxmargingen,
                    scheduler=args.scheduler,
                    train_dis_every_n_iters=args.train_dis_every_n_iters,
                    train_gen_every_n_iters=args.train_gen_every_n_iters,
                    discriminator_train_amount=args.discriminator_train_amount,
                    generator_train_amount=args.generator_train_amount,
                    confounderdis=args.confounderdis,
                    batch_size=batch_size,
                    add_critic_noise=args.add_critic_noise
                    )
    print("Creating the logger")
    logger = WandbLogger(project="lm_gan", config=args)
    # logger = None

    print("Initializing trainer")
    checkpoint = ModelCheckpoint(dirpath=args.output_folder + "/lm_gan" + ("_mem" if args.use_mem else "") + "/",
                                 filename="lmg",
                                 verbose=True,
                                 every_n_train_steps=500)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # trainer = pl.Trainer(gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
    trainer = pl.Trainer(gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
                         logger=logger,
                         max_epochs=args.epochs,
                         precision=16 if args.fp16 else 32,
                         plugins=args.plugins,
                         log_every_n_steps=args.log_every_n_steps,
                         num_processes=args.num_processes,
                         callbacks=[checkpoint, lr_monitor],
                         strategy=args.accelerator,
                         # resume_from_checkpoint=args.resumepath
                         )
    print(trainer.world_size, trainer.local_rank, trainer.global_rank, trainer.node_rank)
    print(trainer.training_type_plugin)
    print("Trainer:")
    print(trainer)
    print("Fitting")
    if args.resumepath is not None:
        print("Loading from checkpoint",args.resumepath)
        model = model.load_from_checkpoint(args.resumepath)
    # trainer.test(model,test_loader)
    if args.do_train:
        trainer.fit(model, train_loader)
        trainer.save_checkpoint(
            args.output_folder + "/lm_gan" + ("_mem" if args.use_mem else "") + "/" + "final_verification.ckpt")
    if args.do_test:
        trainer.test(model, test_loader)
