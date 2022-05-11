from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def _get_model_outputs(model, key):
    if key == 'multihead_output':
        # get list (layers) of multihead module outputs
        return [layer.attention.self.multihead_output for layer in model.encoder.layer]
    elif key == 'layer_output':
        # get list of encoder LayerNorm layer outputs
        return [layer.output.layer_output for layer in model.encoder.layer]
    else:
        raise ValueError("Key not found: %s" % (key))



def compute_Fisher(model, input_mask, total_tokens):
    outputs = {}

    base_model = model.roberta
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad
            if score is not None and name not in outputs:
                score = score ** 2.0
                outputs[name] = score
    # activations
    for key in ['multihead_output', 'layer_output']:
        model_outputs = _get_model_outputs(base_model, key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = 'encoder.layer.{}.{}'.format(i, key)
            model_outputs_i = model_outputs[i].grad

            if model_outputs_i is not None:
                score = torch.einsum("ijk,ij->ijk", [model_outputs_i,   # batch_size x max_seq_length x hidden_size
                                                 input_mask.float()])   # batch_size x max_seq_length
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score ** 2.0
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score
    # cls output
    name = 'cls_output'
    dH = model.classifier.cls_output.grad # This one uses final output of FC layer, size of labels
    w = model.classifier.out_proj.weight
    dx = torch.matmul(dH, w) # To implicate original paper, I multiplied W vector of FC layer (It's same as calculating grad of x)
    score = dx

    if score is not None and name not in outputs:
        score = score.sum(0)
        score = score ** 2.0
        # normalize
        score = score / total_tokens
        outputs[name] = score

    # task-specific layer
    for name, parameter in model.named_parameters():
        if 'roberta' not in name:
            score = parameter.grad
            if score is not None and name not in outputs:
                score = score ** 2.0
                outputs[name] = score

    return outputs

def compute_Fisher_with_labels(model, input_mask, loss):
    total_tokens = input_mask.float().detach().sum().data

    model.zero_grad()
    loss.backward()
    outputs = compute_Fisher(model, input_mask, total_tokens)
    return outputs

def compute_taskemb(dataloader, model, taskname):
    model.zero_grad()
    global_feature_dict = {}
    num_examples = 0
    total_num_examples = 0

    batched = []
    from tqdm import tqdm
    print('loading data into GPU...')
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        batched.append(batch)

    print('Calculating on GPU...')
    for batch in tqdm(batched):
        outputs = model(**batch)
        loss, logits = outputs[0], outputs[1]
        input_mask = batch['attention_mask']
        feature_dict = compute_Fisher_with_labels(model, input_mask, loss)

        if len(global_feature_dict) == 0:
            for key in feature_dict:
                global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
        else:
            for key in feature_dict:
                global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()
            num_examples += batch['input_ids'].size(0)
        total_num_examples += num_examples
    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    # Save features
    import os
    os.makedirs('./taskemb_'+ taskname)
    for key in global_feature_dict:
        np.save(os.path.join('./taskemb_'+ taskname , '{}.npy'.format(key)), global_feature_dict[key])

def run_taskemb(dataset_name, modelpath, taskname):

    dataset = load_dataset('json', data_files={'test': dataset_name},
                           field='data')
    test_dataset = dataset["test"].shuffle(seed=42)
    model = AutoModelForSequenceClassification.from_pretrained(modelpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model.to(device)

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    test_dataset = test_dataset.map(lambda x : tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    test_dataset.set_format(type='torch', columns=['labels', 'input_ids', 'attention_mask'])
    dataloader = DataLoader(test_dataset, batch_size=24, num_workers=8, pin_memory=True)
    compute_taskemb(dataloader, model, taskname)


def main():
    run_taskemb(
        dataset_name='./data/labeled_sst/labeled3_sst_train.json',
        modelpath='./checkpoints/sentiment/senti_sst_labeled3/best-loss',
        taskname='SST'
    )
    run_taskemb(
        dataset_name='./data/labeled_imdb/labeled_imdb_train.json',
        modelpath='./checkpoints/sentiment/senti_imdb/best-loss',
        taskname='imdb'
    )
    run_taskemb(
        dataset_name='./data/labeled_cola/labeled_cola_train.json',
        modelpath='./checkpoints/another_task/cola/best-loss',
        taskname='cola'
    )
    run_taskemb(
        dataset_name='./data/labeled_agnews/labeled_agnews_train.json',
        modelpath='./checkpoints/another_task/agnews/best-loss',
        taskname='agnews'
    )


if __name__ == "__main__":
    main()
