import torch
from tokenizers import Tokenizer
from DOME_dataloader import DOMEDataset
from torch.utils.data import DataLoader
from DOME import Generator
from utils import eval_bleu_rouge_meteor
from tqdm import tqdm

if __name__ == '__main__':
    modelParametersDict = {
        "numberOfEncoderLayers": 4,
        "numberOfDecoderLayers": 4,
        "ffDimension": 1024,
        "modelDimension" : 256,
        "intentDimension": 128,
        "intentNum":5,
        "attentionHeadsNum": 8,
        "maxLengthoOfComment": 30,
        "maxLineNum":15,
        "clipDistCode": 8,
        "maxTokenInline": 25,
        "statK":5,
        "tokenK":10,
        "BeamWidth": 5,
        "dropoutRate": 0.2
    }

    dataset = 'funcom'
    tokenizer = Tokenizer.from_file('./dataset/funcom/bpe_tokenizer_all_token.json')
    batchSize= 32

    testingSet = DOMEDataset(tokenizer=tokenizer, dataset=dataset, mode='test',
                           max_token_inline=modelParametersDict["maxTokenInline"], 
                            max_line_num=modelParametersDict["maxLineNum"], max_comment_len=modelParametersDict["maxLengthoOfComment"])
    
    vocabSize = tokenizer.get_vocab_size()
    eosToken = tokenizer.token_to_id('[EOS]')

    testLoader = DataLoader(testingSet,
                             batch_size=batchSize,
                             collate_fn=testingSet.collate_fn,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=False)
    model = Generator(modelParametersDict["modelDimension"], modelParametersDict["intentDimension"], modelParametersDict["ffDimension"], 
                      modelParametersDict["attentionHeadsNum"], modelParametersDict["numberOfEncoderLayers"], modelParametersDict["numberOfDecoderLayers"], 
                      vocabSize,
                      modelParametersDict["maxLengthoOfComment"], modelParametersDict["clipDistCode"], eosToken,
                      modelParametersDict["intentNum"], modelParametersDict["statK"], modelParametersDict["tokenK"], modelParametersDict["dropoutRate"], None)
    
    
    dataset = 'funcom'
    model.load_state_dict(torch.load(f"./saved_model/funcom/comment_generator.pkl"))
    model.cuda()
    model.eval()
    ids, intents, commentPrediction, commentReference = [], [], [], []
    with torch.no_grad():
        for t in tqdm(testLoader):
            codeId_test = t[len(t)-1]
            comment_test = t[3]
            code_test, validCodeLength_test, exemplar_test = t[0].cuda(), t[1].cuda(), t[2].cuda()
            validExemplarLength_test, intent_test  = t[4].cuda(), t[6].cuda() 
            bos_test = t[7].cuda().reshape(-1, 1)
            predictedComment_test = model(code_test, exemplar_test, bos_test, validCodeLength_test, validExemplarLength_test, intent_test)
            j = 0
            while j < len(comment_test):
                ref = comment_test[j]
                commentReference.append([ref])
                pred = tokenizer.decode(predictedComment_test[j]).split()
                if not pred:
                    commentPrediction.append(['1'])
                else:
                    commentPrediction.append(pred)
                j += 1
            ids += codeId_test
            intents += intent_test.tolist()
    
    idtoIntent = ['what', 'why', 'usage', 'done', 'property']
    intentId = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    intentReference = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    intentPredicted = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    
    for ii, pp, rr, ll in zip(ids, commentPrediction, commentReference, intents):
        ll = idtoIntent[ll]
        intentId[ll].append(ii)
        intentPredicted[ll].append(pp)
        intentReference[ll].append(rr)
    bleuOfWhat, rougeOfWhat, meteorOfWhat = eval_bleu_rouge_meteor(intentId['what'], intentPredicted['what'], intentReference['what'])[:3]
    bleuOfWhy, rougeOfWhy, meteorOfWhy = eval_bleu_rouge_meteor(intentId['why'], intentPredicted['why'], intentReference['why'])[:3]
    bleuOfUsage, rougeOfUsage, meteorOfUsage = eval_bleu_rouge_meteor(intentId['usage'], intentPredicted['usage'], intentReference['usage'])[:3]
    bleuOfDone, rougeOfDone, meteorOfDone = eval_bleu_rouge_meteor(intentId['done'], intentPredicted['done'], intentReference['done'])[:3]
    bleuOfProperty, rougeOfProperty, meteorOfProperty = eval_bleu_rouge_meteor(intentId['property'], intentPredicted['property'], intentReference['property'])[:3]

    print("Results on Testing Set: ")
    print("For What: "+ "BLEU =" + bleuOfWhat + "ROUGE =" + rougeOfWhat + "METEOR =" + meteorOfWhat )
    print("For Why: "+ "BLEU =" + bleuOfWhy + "ROUGE =" + rougeOfWhy + "METEOR =" + meteorOfWhy )
    print("For Usage: "+ "BLEU =" + bleuOfUsage + "ROUGE =" + rougeOfUsage + "METEOR =" + meteorOfUsage )
    print("For Done: "+ "BLEU =" + bleuOfDone + "ROUGE =" + rougeOfDone + "METEOR =" + meteorOfDone )
    print("For Property: "+ "BLEU =" + bleuOfProperty + "ROUGE =" + rougeOfProperty + "METEOR =" + meteorOfProperty )

    with open('DOME.txt', 'w') as w:
        for c in commentPrediction:
            comment = ' '.join(c)
            w.write(comment + '\n')
    print("Completed")

