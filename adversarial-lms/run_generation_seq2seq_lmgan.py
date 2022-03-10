import datetime
import sys

import nltk
import torch.cuda
from nltk import sent_tokenize

from model.kbart.kgcbartgannormalgan import BartGAN


def generate_sequence(input_text, model, tokenizer, device, top_k=155,p=False,max_length=24,num_beams=1,sample=False):
    print("Generating using device",device)
    if not isinstance(input_text,list):
        input_text = [input_text]
    s = datetime.datetime.now()
    source = tokenizer(input_text, max_length=max_length, pad_to_max_length=True,
                                         return_tensors='pt')
    source_ids = source['input_ids']  # .unsqueeze()
    source_mask = source['attention_mask']  # .unsqueeze()

    if device is not None:
        source_ids = source_ids.to(device)  # .unsqueeze()
        source_mask = source_mask.to(device)  # .unsqueeze()

    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=source_mask,
        max_length=max_length,
        num_beams=num_beams,
        do_sample=sample,
        top_k=top_k
    )
    f = datetime.datetime.now()
    print(f-s)
    res = []
    for i in range(generated_ids.shape[0]):
        gids = generated_ids[i, :].tolist()
        s = tokenizer.decode(gids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        if s not in res:
            res.append(s)
        if p:
            print(s)
    return res



if __name__ == '__main__':
    model_path = sys.argv[1]
    print("Loading",model_path)
    model = BartGAN.load_from_checkpoint(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # sentences = ["You are in the kitchen and you have one potato.","You the potato on the table.","You walk to the basement.",
    #              "Now you are in the basement. ","You step outside.","You go for a walk."]
    stry = "I'll begin from the moment I got diary, the moment I saw diary lying on the table among my other birthday presents. (I went along when diary were bought, but that doesn't count.) " \
           "On Friday, June 12, I was awake at six o'clock, which isn't surprising, since it was my birthday." \
           # "I made a mistake writing the date. Check to see if you made a fixable mistake like a " \
           # "misspelled name or wrong date. Cross out the mistake and write the correction on the check. " \
           # "Write your initials next to the corrected mistake. "
    # stry = "The hockey game was tied up. The red team against the blue team. The red team had the puck.  They sprinted down the ice. They cracked a shot on goal!. They scored a final goal!."
    # stry = "The hockey game was tied up. The red team had the puck.  They sprinted down the ice. They cracked a shot on goal!. They scored a final goal!."
    # stry = "The hockey game was tied up. The red team had the puck. They sprinted down the ice. They cracked a shot on goal!. They missed!."
    # stry = "Alex wanted to cook a steak. Alex turned on the stove. Alex put the steak in the pan. Alex put the pan on the stove. Alex stepped out for a while."
    # stry = 'Alex needed a t-shirt. He thought he could get it at the mall. Alex goes to the mall with many people. The lines were insane.'
    sentences = nltk.sent_tokenize(stry)
    ctext = "<story> "+ " ".join(sentences)
    # for keyword, score in kw_model.extract_keywords(" ".join(sentences)):
    for s in sentences:
        # ts = "<sentence> "+s+"( <subj> "+keyword+")"
        ts = "<sentence> "+s+"( <general> <subject> diary <relation> located at  )"
        tctext = ctext+ts

        print("Input:")
        print(tctext)
        res = model.generate_and_eval([tctext],max_length=1024,num_return_sequences=10)
        for tup in sorted([(fact,key) for fact,key in zip(res[0],res[1])],key=lambda x: x[1],reverse=True):
            print(tup)
        print("*"*10)
