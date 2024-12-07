import torch
from transformers import AutoTokenizer
import string
from model import commentClassifier  # Ensure these are correctly imported

def coin_preprocess(tokenizer, comment):
    def count_punc_num(comment, comment_len):
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        punc_num = count(comment, set(string.punctuation))
        digits_num = count(comment, set(string.digits))
        return (punc_num + digits_num) / comment_len

    comment_tokens = tokenizer.tokenize(comment)
    cc_tokens = [tokenizer.cls_token] + comment_tokens + [tokenizer.sep_token]
    cc_ids = tokenizer.convert_tokens_to_ids(cc_tokens)
    cc_att_mask = [1] * len(cc_tokens)
    punc_num = count_punc_num(comment, len(comment.strip().split()))
    if len(comment.strip().split()) < 3:
        comment_len = 1
    else:
        comment_len = 0    
    return torch.tensor(cc_ids).unsqueeze(0), torch.tensor(cc_att_mask).unsqueeze(0), \
           torch.tensor(comment_len).unsqueeze(0), torch.tensor(punc_num).unsqueeze(0)\


def main():
    pretrained_model_path = 'microsoft/codebert-base'
    saved_model_path = 'C:\\Users\\krush\\Downloads\\NLP-main\\NLP_main\\src\\comment_classifier\\trained_models\\model_fold_9.pth'
    classifier = commentClassifier(pretrained_model_path, 6, 0.2)
    classifier.load_state_dict(torch.load(saved_model_path))
    classifier.cuda()
    print("load the parameters of the pretrained classifier!")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    class_name = ['what', 'why', 'how-to-use', 'how-it-is-done', 'property', 'others']
    # comment = 'function getCondition() is required to exit the loop'
    comment = 'loop breaks if condition is false'
    input_ids, attention_mask, comment_len, punc_num = coin_preprocess(tokenizer, comment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    comment_len = comment_len.to(device)
    punc_num = punc_num.to(device)
    classifier.eval()
    # inputs = coin_preprocess(tokenizer, comment)
    # inputs = {key: value.cuda() for key, value in inputs.items()}  # Move inputs to GPU
    # classifier.eval()
    with torch.no_grad():
        # logits = classifier(**inputs)
        logits = classifier(input_ids, attention_mask, comment_len, punc_num)
        intent = class_name[int(torch.argmax(logits, dim=1))]
    print('Comment:', comment)
    print('Intent:', intent)


if __name__ == "__main__":
    main()
