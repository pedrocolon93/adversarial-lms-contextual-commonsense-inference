import datetime
import sys

import torch.cuda
from nltk import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration, LogitsProcessorList, MinLengthLogitsProcessor

from model.kbart.kgcbartgan import BartGAN


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
    # tokenizer = model.generator_tok
    # model = model.generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ctext = "<story> It was raining all day. " \
    #         "<sentence> It was raining all day. ( <relation> is located at) "

    # ctext = "<story> You are in the basement. " \
    #         "<sentence> A potato. ( <relation> located at <subj> potato ) "
    # ctext = "<story> The hockey game was tied up. The blue team had the puck. They sprinted down the ice. They cracked a shot on goal!. They scored it!. <sentence> The hockey game was tied up."

    # "relation":"<specific> <relation> is capable of <subj> A cat <obj> drink milk <\/relation>" "
    # ctext = "<story> " \
    #         "<sentence> The tea tasted like syrup. ( <general> )"
    print("Input:")
    sentences = ["You have a potato.","You are in the kitchen.",
                 "You put a potato on the table.", \
            "Now you are in the basement.","Now you are walking outside. "]
    # sentences = sent_tokenize(str(input("Give me a couple of sentences for a story.")))
    ctext = ' '.join(sentences)
    print(ctext)

    hint = str(input("Now give me a hint."))
    threshold = float(input("Give me a threshold from 0-1 (Default:0.2)",) or 0.2)

    # hint = "( <relation> located at, <subj> potato )"
    print("Hint:")
    print(hint)


    # res = generate_sequence(ctext,model,tokenizer,device,p=True,max_length=256)

    update_mem = None
    for sentence in sentences:
        inpt = ctext+sentence+" "+hint
        res = model.generate_and_eval([inpt],update_mem=update_mem,num_return_sequences=2,num_beams=10)
        update_mem = [res[0][x].replace("</s>","").replace("<s>","") for x in range(len(res[0])) if res[1][x][0]>threshold ]
        print("Sentence",sentence, res[0], res[1])
        if len(update_mem) == 0:
            update_mem = None
        print("memory update",update_mem)
    # ctext = "<story> Step one: Write the date on the line in the upper right-hand corner. Step two: Write the name of the recipient. " \
    #         "Step three: Write the amount of the check to the right of the dollar sign. " \
    #         "Step four: Write the monetary amount of the check in word form below the \"Pay to the Order of\" line. " \
    #         "Step five: Sign the check on the line in the bottom right corner. " \
    #         "Step six: Fill out the memo section on the bottom left of the check." \
    #         "<sentence> Step five: Sign the check on the line in the bottom right corner. ( <relation> before )"

