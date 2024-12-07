from DOME import Generator
from tokenizers import Tokenizer
import numpy as np
import torch
import json

class Config(object):
    def __init__(self):
        self.bpe_model = f'C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\Application_Demo\\bpe_tokenizer_all_token.json'
        self.tokenizer = Tokenizer.from_file(self.bpe_model)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_token = self.tokenizer.token_to_id('[EOS]')
        self.intent2id = {'what': 0, 'why': 1, 'usage': 2, 'done': 3, 'property': 4}
        self.intent2bos_id = {'what': "[WHAT/]", 'why': "[WHY/]", 'usage': "[USAGE/]", 'done': "[DONE/]", 'property': "[PROP/]"}
        self.intent2cls_id = {'what': "[/WHAT]", 'why': "[/WHY]", 'usage': "[/USAGE]", 'done': "[/DONE]", 'property': "[/PROP]"}

        self.d_model = 512
        self.d_intent = 128
        self.d_ff = 2048
        self.head_num = 8
        self.enc_layer_num = 6
        self.dec_layer_num = 6
        self.max_token_inline = 25
        self.max_line_num = 15
        self.max_comment_len = 30
        self.clip_dist_code = 8
        self.intent_num = 5
        self.stat_k = 5
        self.token_k = 10
        self.beam_width = 5
        self.batch_size = 64
        self.dropout = 0.2
    
config = Config()
generator = Generator(config.d_model, config.d_intent, config.d_ff, config.head_num, config.enc_layer_num,
                      config.dec_layer_num, config.vocab_size, config.max_comment_len, config.clip_dist_code, config.eos_token,
                      config.intent_num, config.stat_k, config.token_k, config.dropout, None)
generator.load_state_dict(torch.load(f"C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\comment_generator\\saved_model\\funcom\\comment_generator.pkl"))
generator.cuda()
print("load the parameters of the pretrained generator!")

with open('C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\Application_Demo\\demo_generator_dataset\\raw_code.demo', 'r') as f:
    raw_code_lines = f.readlines()
with open('C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\Application_Demo\\demo_generator_dataset\\code_split.demo', 'r') as f:
    code_stat_lines = f.readlines()
with open('C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\Application_Demo\\demo_generator_dataset\\similar_comment.demo', 'r') as f:
    similar_comment_lines = f.readlines()

raw_code, input_code, code_valid_len, input_exemplar = [], [], [], []
for raw_code_line, code_stat_line, exemplar_line in zip(raw_code_lines, code_stat_lines, similar_comment_lines):
    raw_code.append(json.loads(raw_code_line.strip())['raw_code'])
    statement_line = json.loads(code_stat_line.strip())
    exemplar_what = json.loads(exemplar_line.strip())['what']
    exemplar_why = json.loads(exemplar_line.strip())['why']
    exemplar_done = json.loads(exemplar_line.strip())['done']
    exemplar_usage = json.loads(exemplar_line.strip())['usage']
    exemplar_property = json.loads(exemplar_line.strip())['property']
    input_exemplar.append({'what':config.tokenizer.encode(exemplar_what).ids[:config.max_comment_len], 'why':config.tokenizer.encode(exemplar_why).ids[:config.max_comment_len], 'done':config.tokenizer.encode(exemplar_done).ids[:config.max_comment_len], 'usage':config.tokenizer.encode(exemplar_usage).ids[:config.max_comment_len], 'property':config.tokenizer.encode(exemplar_property).ids[:config.max_comment_len]})
    
    temp_code = []
    for stat_idx, stat in enumerate(statement_line['code'][:config.max_line_num]):
        cur_stat = config.tokenizer.encode(stat).ids[:config.max_token_inline]
        temp_code = temp_code + cur_stat + [config.tokenizer.token_to_id('[PAD]')] * (config.max_token_inline - len(cur_stat))
    input_code.append(temp_code)
    
    code_valid_len.append(len(statement_line['code'][:config.max_line_num]))

def prediction(code, exemplar, intent, code_valid_len):
    input_intent = torch.tensor(config.intent2id[intent]).unsqueeze(0).cuda()
    bos = torch.tensor([config.tokenizer.token_to_id(config.intent2bos_id[intent])]).unsqueeze(0).cuda()
    input_code = torch.tensor([config.tokenizer.token_to_id(config.intent2cls_id[intent])] + code).unsqueeze(0).cuda()
    input_exemplar = torch.tensor(exemplar[intent]).unsqueeze(0).cuda()
    code_valid_len = torch.tensor([code_valid_len]).cuda()
    exemplar_valid_len = torch.tensor([len(exemplar[intent])]).cuda()
    generator.eval()
    pred = generator(input_code, input_exemplar, bos, code_valid_len, exemplar_valid_len, input_intent)
    pred = config.tokenizer.decode(pred[0])
    return pred

    # 3.prediction
for i in range(len(input_code)):
    print("code:\n", raw_code[i])
    print("what:", prediction(input_code[i], input_exemplar[i], 'what', code_valid_len[i]))
    print("why:", prediction(input_code[i], input_exemplar[i], 'why', code_valid_len[i]))
    print("how-it-is-done:", prediction(input_code[i], input_exemplar[i], 'done', code_valid_len[i]))
    print("usage:", prediction(input_code[i], input_exemplar[i], 'usage', code_valid_len[i]))
    print("property:", prediction(input_code[i], input_exemplar[i], 'property', code_valid_len[i]))
    print("=============================================================================")