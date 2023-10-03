import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast,BertModel
from tqdm import tqdm
from pprint import pprint
from utils import Preprocessor, DefaultLogger,MyDataset
from OD_RTE import ODRTEDataMaker4Bert,MetricsCalculator_ODRTE,ODRTETaggingScheme
from OD_RTE import RelModel_OD_RTE
import time
import glob
import logging


seed = 999
max_seq_len = 100
sliding_len = 20
batch_size = 6
epochs = 250#10-20 for nyt and nyt_star
init_learning_rate = 5e-5
rel_neg_sample_num = 20
log_interval = 300
f1_2_save = 0
T_mult = 2
global_retion_loss_factor = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scheduler = "CAWR"
bert_dir = "pretrained_models/bert-base-cased"
data_dir = "./data/webnlg"#webnlg,webnlg_star,nyt,nyt_star
log_path = "./default_log_dir/log.log"
experiment_name = "experiment"
run_name = "OD-RTE"
run_id = 0

logger = DefaultLogger(log_path, experiment_name, run_name, run_id)


torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


train_data_path = os.path.join(data_dir,"train_data.json")
valid_data_path = os.path.join(data_dir,"test_data.json")
rel2id_path = os.path.join(data_dir, "rel2id.json")
train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))
rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))

train_data = train_data
valid_data = valid_data



tokenizer = BertTokenizerFast.from_pretrained(bert_dir, add_special_tokens = False, do_lower_case = False)
tokenize = tokenizer.tokenize
handshaking_tagger = ODRTETaggingScheme(rel2id = rel2id, max_seq_len = max_seq_len)
data_maker = ODRTEDataMaker4Bert(tokenizer, handshaking_tagger,rel_neg_sample_num)
get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]
preprocessor = Preprocessor(tokenize_func = tokenize, get_tok2char_span_map_func = get_tok2char_span_map)






# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data 
    
for sample in all_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))


print("The maximum token length of a sample in the dataset is {}".format(max_tok_num))

if max_tok_num > max_seq_len:
    train_data = preprocessor.split_into_short_samples(train_data, max_seq_len, sliding_len = sliding_len, encoder = "BERT")
    valid_data = preprocessor.split_into_short_samples(valid_data, max_seq_len, sliding_len = sliding_len, encoder = "BERT")

print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))

indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)


train_dataloader = DataLoader(MyDataset(indexed_train_data), 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  pin_memory=True,
                                  num_workers = 2,
                                  drop_last = False,
                                  collate_fn = data_maker.generate_batch_train,
                                 )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data), 
                          batch_size = 6, 
                          shuffle = True, 
                          num_workers = 2,
                          drop_last = False,
                          collate_fn = data_maker.generate_batch_valid,
                         )

bert = BertModel.from_pretrained(bert_dir)

rel_extractor = RelModel_OD_RTE(bert,len(rel2id),len(handshaking_tagger.id2tag),max_seq_len,device)

rel_extractor = rel_extractor.to(device)



optimizer = torch.optim.Adam(rel_extractor.parameters(), lr = init_learning_rate)
if scheduler == "CAWR":
    
    rewarm_epoch_num = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    
elif scheduler == "Step":
    decay_rate = 0.999
    decay_steps = 100
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_steps, gamma = decay_rate)


loss_func = nn.BCEWithLogitsLoss()

metrics = MetricsCalculator_ODRTE(handshaking_tagger)


def train_step(batch_train_data, optimizer):
    sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_shaking_tag,batch_rel_potential_vec,batch_rel_potential_list_list,batch_sep_index_list,global_region_tag = batch_train_data
    
    batch_input_ids,batch_attention_mask,batch_token_type_ids,batch_shaking_tag,batch_rel_potential_vec,global_region_tag = (batch_input_ids.to(device), 
                                    batch_attention_mask.to(device), 
                                    batch_token_type_ids.to(device), 
                                    batch_shaking_tag.to(device),
                                    batch_rel_potential_vec.to(device),
                                    global_region_tag.to(device)
                                    )
    batch_rel_potential_list_list = [rel_potential_list_list.to(device) for rel_potential_list_list in batch_rel_potential_list_list]
    

    # zero the parameter gradients
    optimizer.zero_grad()
    tag_outputs,global_region_scores = rel_extractor(batch_input_ids, 
                                                batch_attention_mask, 
                                                batch_token_type_ids, 
                                                batch_rel_potential_vec,
                                                batch_rel_potential_list_list,
                                                batch_sep_index_list,
                                                is_Train = True
                                                )
    

    loss = loss_func(tag_outputs,batch_shaking_tag)+global_retion_loss_factor*loss_func(global_region_scores.reshape(-1),global_region_tag.reshape(-1))
    #loss = loss_func( torch.cat((tag_outputs.reshape(-1),global_region_scores.reshape(-1)),0),torch.cat((batch_shaking_tag.reshape(-1),global_region_tag.reshape(-1)),0)    )
    loss.backward()
    optimizer.step()
    
    
    return loss.item()

# valid step
def valid_step(batch_valid_data):
    sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_shaking_tag,batch_rel_potential_vec,batch_rel_potential_list_list,batch_sep_index_list = batch_valid_data
    
    batch_input_ids,batch_attention_mask,batch_token_type_ids,batch_shaking_tag,batch_rel_potential_vec = (batch_input_ids.to(device), 
                                    batch_attention_mask.to(device), 
                                    batch_token_type_ids.to(device), 
                                    batch_shaking_tag.to(device),
                                    batch_rel_potential_vec.to(device)
                                    )
    batch_rel_potential_list_list = [rel_potential_list_list.to(device) for rel_potential_list_list in batch_rel_potential_list_list]
    
    with torch.no_grad():
        tag_outputs = rel_extractor(batch_input_ids, 
                                                    batch_attention_mask, 
                                                    batch_token_type_ids, 
                                                    batch_rel_potential_vec,
                                                    batch_rel_potential_list_list,
                                                    batch_sep_index_list,
                                                    is_Train = False
                                                    )
    rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list, tag_outputs)
    
    return rel_cpg


max_f1 = 0.
def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):  
    def train(dataloader, ep):
        # train
        rel_extractor.train()
        
        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss = 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(dataloader)
            current_step = steps_per_ep * ep + batch_ind
            
            loss = train_step(batch_train_data, optimizer)
            scheduler.step()
            
            total_loss += loss
            
            avg_loss = total_loss / (batch_ind + 1)

            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}."
            print(batch_print_format.format(experiment_name, run_name, ep + 1, num_epoch, batch_ind + 1, len(dataloader), avg_loss))

            
            if batch_ind % log_interval == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })
    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()
        
        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc,total_rel_sample_acc = 0., 0., 0.,0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        global f1_max_temp
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc = "Validating")):
            rel_cpg = valid_step(batch_valid_data)
            
            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]
        
        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)
        log_dict = {
                        "val_prec": rel_prf[0],
                        "val_recall": rel_prf[1],
                        "val_f1": rel_prf[2],
                        "epoch":ep,
                        "time": time.time() - t_ep,
                        "total_rel_correct_num":total_rel_correct_num,
                        "total_rel_pred_num":total_rel_pred_num,
                        "total_rel_gold_num":total_rel_gold_num

                    }
        logger.log(log_dict)
        pprint(log_dict)
        
        return rel_prf[2]
    
    for ep in range(num_epoch):
        train(train_dataloader, ep)
        if ep>-1:   
            valid_f1 = valid(valid_dataloader, ep)
            
            global max_f1
            if valid_f1 >= max_f1: 
                max_f1 = valid_f1
                log_dict = {"best_f1":max_f1}
                logger.log(log_dict)
            #     if valid_f1 > f1_2_save: # save the best model
            #         modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
            #         torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
    #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
    #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num))) 
            print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, epochs)