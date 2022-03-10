from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer, T5Config, \
    AutoModelForSeq2SeqLM

if __name__ == '__main__':
    path = "t5hint2"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    config = T5Config.from_pretrained("t5-base")
    add_toks = ["<|subj|>", "<|rel|>", "<|obj|>", "<|general|>", "<|specific|>"]
    special_tokens_dict = {'additional_special_tokens': add_toks}
    tokenizer.add_special_tokens(special_tokens_dict)
    config.vocab_size = len(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(
        "./"+path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    for i in [6]:
        print("The dimension is",i+1,"\n")
        input = ": The hockey game was tied up. The red team was against the blue team. " \
                "The red team had the puck. They sprinted down the ice. They cracked a shot on goal!. " \
                "* They scored a final goal!. *" \
                ""
        hints = ["","hint: (<|specific|><|subj|> the red team scores the final goal)",
                 "hint: (<|specific|><|subj|> the blue team does not score the final goal)",
                 "hint: (<|specific|><|obj|> a child)",
                 "hint: (<|general|><|subj|> Something_A (that is a point))"]
        for hint in hints:
            input2 = str(i+1)+input+hint
            print("The input is:")
            print(input2)
            XMB = tokenizer(input2,return_tensors='pt')
            num_beams = 10
            res = model.generate(input_ids=XMB["input_ids"],
                           num_beams=num_beams,
                           num_return_sequences=2,
                           max_length =70)
            for x in res:
                dec = tokenizer.decode(x,skip_special_tokens=True)
                print(dec)
