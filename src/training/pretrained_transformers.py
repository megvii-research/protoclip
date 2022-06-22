import torch
from pytorch_transformers import RobertaModel, RobertaTokenizer

def get_pretrained_text_encoder_and_tokenizer(name):
    # TODO: get the width of other models
    MODELS = [
        # (BertModel,       BertTokenizer,      'bert-base-uncased', ?),
        # (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt', ?),
        # (GPT2Model,       GPT2Tokenizer,      'gpt2', ?),
        # (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103', ?),
        # (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased', ?),
        # (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024', ?),
        (RobertaModel,    RobertaTokenizer,   'roberta-base', 768)
        ]

    for model_class, tokenizer_class, pretrained_weights, feature_dim in MODELS:
        if pretrained_weights==name:
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights, max_len=75)
            model = model_class.from_pretrained(pretrained_weights)
            
            def tokenize(text):
                result = torch.zeros(77)
                token = torch.tensor(tokenizer.encode(text, add_special_tokens=True))[:77]
                result[:len(token)] = token
                return result.long()

            return model, tokenize, feature_dim


if __name__=='__main__':
    MODELS = [(RobertaModel,    RobertaTokenizer,   'roberta-base')]
    # Let's encode some text in a sequence of hidden-states using each model:
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        print(model)
        # Encode text
        input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        print(input_ids)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
            print(last_hidden_states.size())
