import re
from tkinter import N
from tqdm import tqdm
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json
from torch.nn.parameter import Parameter
from utils import HandshakingKernel
import math
import random

class ODRTETaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(ODRTETaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}
        self.tag2id_onerel = {
            "HB-TB":0,
            "HB-TE":1,
            "HE-TE":2,
            "HE-TB":3,
        }
        self.id2tag = {id_:tag for tag, id_ in self.tag2id_onerel.items()}

        self.matrix_size = max_seq_len

    def get_spots(self, sample):

        hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots = [], [], []
        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            hb_tb_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[0],self.tag2id_onerel["HB-TB"]))
            hb_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[1] - 1,self.tag2id_onerel["HB-TE"]))
            he_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[1] - 1,1+obj_tok_span[1] - 1,self.tag2id_onerel["HE-TE"]))
                
        return hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots

    def spots2batch_tag(self,hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
        batch_size = len(hb_tb_matrix_spots)
        batch_tag = torch.zeros(batch_size,len(self.rel2id),self.matrix_size+2,self.matrix_size+2,len(self.tag2id_onerel))
        global_region_tag = torch.zeros(batch_size,self.matrix_size+2,self.matrix_size+2)
        batch_ind = 0
        for hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i in zip(hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
            for hb_tb_matrix_spots_ij,hb_te_matrix_spots_ij,he_te_matrix_spots_ij in zip(hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i):
                batch_tag[(batch_ind,)+hb_tb_matrix_spots_ij] = 1

                batch_tag[(batch_ind,)+hb_te_matrix_spots_ij] = 1

                batch_tag[(batch_ind,)+he_te_matrix_spots_ij] = 1
                rel_id_temp = hb_tb_matrix_spots_ij[0]
                hb_id = hb_tb_matrix_spots_ij[1]
                he_id = he_te_matrix_spots_ij[1]
                tb_id = hb_tb_matrix_spots_ij[2]
                te_id = he_te_matrix_spots_ij[2]
                batch_tag[batch_ind,rel_id_temp,he_id,tb_id,self.tag2id_onerel["HE-TB"]]=1
                global_region_tag[batch_ind,hb_id:he_id+1,tb_id:te_id+1]=1

            batch_ind+=1
        return batch_tag,global_region_tag


    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      text, 
                      shaking_tag,
                      tok2char_span, 
                      tok_offset = 0, char_offset = 0):

        
        rel_list1 = []
        # shaking_tag_negvalue = shaking_tag[:,:,:,3,None].expand(-1,-1,-1,3)
        # shaking_tag = (shaking_tag[:,:,:,:-1]>shaking_tag_negvalue).long()
        shaking_tag_temp = shaking_tag

        shaking_tag = (shaking_tag[:,:,:,:-1]>0.5).long()
        rel_num,seq_lens, seq_lens,tag_num = shaking_tag.shape
        for spot_ind,spot in enumerate(shaking_tag.nonzero().tolist()):
            r_index = spot[0]
            h_start_index = spot[1]
            t_start_index = spot[2]
            tag = spot[3]
            if tag == self.tag2id_onerel['HB-TB'] and (spot_ind+1) < len(shaking_tag.nonzero().tolist()):
                if shaking_tag.nonzero().tolist()[spot_ind+1][3] == self.tag2id_onerel['HB-TE']:
                    t_end_index = shaking_tag.nonzero().tolist()[spot_ind+1][2]
                    for h_end_index in range(h_start_index, seq_lens):
                        if shaking_tag[r_index][h_end_index][t_end_index][self.tag2id_onerel['HE-TE']] ==1:
                            subj_tok_span = [h_start_index,h_end_index+1]
                            obj_tok_span = [t_start_index,t_end_index+1]
                            subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
                            obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
                            subject = text[subj_char_span[0]:subj_char_span[1]]
                            object = text[obj_char_span[0]:obj_char_span[1]]
                            rel_list1.append({
                                "subject": subject,
                                "object": object,
                                "subj_tok_span": subj_tok_span,
                                "obj_tok_span": obj_tok_span,
                                "subj_char_span": subj_char_span,
                                "obj_char_span": obj_char_span,
                                "predicate": self.id2rel[r_index],
                            })
                            break

        # shaking_tag = shaking_tag_temp
        # shaking_tag = torch.cat([shaking_tag[:,:,:,0][:,:,:,None],shaking_tag[:,:,:,3][:,:,:,None],shaking_tag[:,:,:,2][:,:,:,None]],-1)#0,3,2
        # shaking_tag = (shaking_tag>0.5).long()
        # tag2id_new = {
        #     "HB-TB":0,
        #     "HE-TB":1,
        #     "HE-TE":2,
        # }
        # rel_list2 = []
        # for spot_ind,spot in enumerate(shaking_tag.nonzero().tolist()):
        #     r_index = spot[0]
        #     h_end_index = spot[1]
        #     t_start_index = spot[2]
        #     tag = spot[3]
        #     if tag == tag2id_new['HE-TB'] and (spot_ind+1) < len(shaking_tag.nonzero().tolist()):
        #         if shaking_tag.nonzero().tolist()[spot_ind+1][3] == tag2id_new['HE-TE']:
        #             t_end_index = shaking_tag.nonzero().tolist()[spot_ind+1][2]
        #             for h_start_index in range(h_end_index, 0,-1):
        #                 if shaking_tag[r_index][h_start_index][t_start_index][tag2id_new['HB-TB']] ==1:
        #                     subj_tok_span = [h_start_index,h_end_index+1]
        #                     obj_tok_span = [t_start_index,t_end_index+1]
        #                     subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
        #                     obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
        #                     subject = text[subj_char_span[0]:subj_char_span[1]]
        #                     object = text[obj_char_span[0]:obj_char_span[1]]
        #                     rel_list2.append({
        #                         "subject": subject,
        #                         "object": object,
        #                         "subj_tok_span": subj_tok_span,
        #                         "obj_tok_span": obj_tok_span,
        #                         "subj_char_span": subj_char_span,
        #                         "obj_char_span": obj_char_span,
        #                         "predicate": self.id2rel[r_index],
        #                     })
        #                     break

        shaking_tag = shaking_tag_temp
        shaking_tag = torch.cat([shaking_tag[:,:,:,0][:,:,:,None],shaking_tag[:,:,:,3][:,:,:,None],shaking_tag[:,:,:,2][:,:,:,None]],-1)#0,3,2
        shaking_tag = (shaking_tag>0.5).long()
        shaking_tag = shaking_tag.permute(0,2,1,3)
        tag2id_new = {
            "HB-TB":0,
            "HE-TB":1,
            "HE-TE":2,
        }
        rel_list2 = []
        for spot_ind,spot in enumerate(shaking_tag.nonzero().tolist()):
            r_index = spot[0]
            h_start_index = spot[2]
            t_start_index = spot[1]
            tag = spot[3]
            if tag == tag2id_new['HB-TB'] and (spot_ind+1) < len(shaking_tag.nonzero().tolist()):
                if shaking_tag.nonzero().tolist()[spot_ind+1][3] == tag2id_new['HE-TB']:
                    h_end_index = shaking_tag.nonzero().tolist()[spot_ind+1][2]
                    for t_end_index in range(t_start_index,seq_lens):
                        if shaking_tag[r_index][t_end_index][h_end_index][tag2id_new['HE-TE']] ==1:
                            subj_tok_span = [h_start_index,h_end_index+1]
                            obj_tok_span = [t_start_index,t_end_index+1]
                            subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
                            obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
                            subject = text[subj_char_span[0]:subj_char_span[1]]
                            object = text[obj_char_span[0]:obj_char_span[1]]
                            rel_list2.append({
                                "subject": subject,
                                "object": object,
                                "subj_tok_span": subj_tok_span,
                                "obj_tok_span": obj_tok_span,
                                "subj_char_span": subj_char_span,
                                "obj_char_span": obj_char_span,
                                "predicate": self.id2rel[r_index],
                            })
                            break



        rel_list = [trip for trip in rel_list1+rel_list2 if (trip in rel_list1) and (trip in rel_list2)]
        

        
        return rel_list




class OneRelTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(OneRelTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}
        self.tag2id_onerel = {
            "HB-TB":0,
            "HB-TE":1,
            "HE-TE":2
        }
        self.id2tag = {id_:tag for tag, id_ in self.tag2id_onerel.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):

        hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots = [], [], []
        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            hb_tb_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[0],self.tag2id_onerel["HB-TB"]))
            hb_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[1] - 1,self.tag2id_onerel["HB-TE"]))
            he_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[1] - 1,1+obj_tok_span[1] - 1,self.tag2id_onerel["HE-TE"]))
                
        return hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots

    def spots2batch_tag(self,hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
        batch_size = len(hb_tb_matrix_spots)
        batch_tag = torch.zeros(batch_size,len(self.rel2id),self.matrix_size+2,self.matrix_size+2,len(self.tag2id_onerel))
        batch_ind = 0
        for hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i in zip(hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
            for hb_tb_matrix_spots_ij,hb_te_matrix_spots_ij,he_te_matrix_spots_ij in zip(hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i):
                batch_tag[(batch_ind,)+hb_tb_matrix_spots_ij] = 1
                batch_tag[(batch_ind,)+hb_te_matrix_spots_ij] = 1
                batch_tag[(batch_ind,)+he_te_matrix_spots_ij] = 1
            batch_ind+=1
        return batch_tag


    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      text, 
                      shaking_tag,
                      tok2char_span, 
                      tok_offset = 0, char_offset = 0):

        rel_list = []
        rel_num,seq_lens, seq_lens,tag_num = shaking_tag.shape
        for spot_ind,spot in enumerate(shaking_tag.nonzero().tolist()):
            r_index = spot[0]
            h_start_index = spot[1]
            t_start_index = spot[2]
            tag = spot[3]
            if tag == self.tag2id_onerel['HB-TB'] and (spot_ind+1) < len(shaking_tag.nonzero().tolist()):
                if shaking_tag.nonzero().tolist()[spot_ind+1][3] == self.tag2id_onerel['HB-TE']:
                    t_end_index = shaking_tag.nonzero().tolist()[spot_ind+1][2]
                    for h_end_index in range(h_start_index, seq_lens):
                        if shaking_tag[r_index][h_end_index][t_end_index][self.tag2id_onerel['HE-TE']] ==1:
                            subj_tok_span = [h_start_index,h_end_index+1]
                            obj_tok_span = [t_start_index,t_end_index+1]
                            subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
                            obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
                            subject = text[subj_char_span[0]:subj_char_span[1]]
                            object = text[obj_char_span[0]:obj_char_span[1]]
                            rel_list.append({
                                "subject": subject,
                                "object": object,
                                "subj_tok_span": subj_tok_span,
                                "obj_tok_span": obj_tok_span,
                                "subj_char_span": subj_char_span,
                                "obj_char_span": obj_char_span,
                                "predicate": self.id2rel[r_index],
                            })
                            break
        return rel_list
class OneRelTaggingSchemeBidireDecode(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(OneRelTaggingSchemeBidireDecode, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}
        self.tag2id_onerel = {
            "HB-TB":0,
            "HB-TE":1,
            "HE-TE":2
        }
        self.id2tag = {id_:tag for tag, id_ in self.tag2id_onerel.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):

        hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots = [], [], []
        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            hb_tb_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[0],self.tag2id_onerel["HB-TB"]))
            hb_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[0],1+obj_tok_span[1] - 1,self.tag2id_onerel["HB-TE"]))
            he_te_matrix_spots.append((self.rel2id[rel["predicate"]],1+subj_tok_span[1] - 1,1+obj_tok_span[1] - 1,self.tag2id_onerel["HE-TE"]))
                
        return hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots

    def spots2batch_tag(self,hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
        batch_size = len(hb_tb_matrix_spots)
        batch_tag = torch.zeros(batch_size,len(self.rel2id),self.matrix_size+2,self.matrix_size+2,len(self.tag2id_onerel))
        batch_ind = 0
        for hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i in zip(hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots):
            for hb_tb_matrix_spots_ij,hb_te_matrix_spots_ij,he_te_matrix_spots_ij in zip(hb_tb_matrix_spots_batch_i,hb_te_matrix_spots_batch_i,he_te_matrix_spots_batch_i):
                batch_tag[(batch_ind,)+hb_tb_matrix_spots_ij] = 1
                batch_tag[(batch_ind,)+hb_te_matrix_spots_ij] = 1
                batch_tag[(batch_ind,)+he_te_matrix_spots_ij] = 1
            batch_ind+=1
        return batch_tag


    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      text, 
                      shaking_tag,
                      tok2char_span, 
                      tok_offset = 0, char_offset = 0):

        rel_list = []
        rel_num,seq_lens, seq_lens,tag_num = shaking_tag.shape
        for spot_ind,spot in enumerate(shaking_tag.nonzero().tolist()):
            r_index = spot[0]
            h_start_index = spot[1]
            t_start_index = spot[2]
            tag = spot[3]
            if tag == self.tag2id_onerel['HB-TB']:
                for t_end_index in range(t_start_index, seq_lens):
                    if shaking_tag[r_index][h_start_index][t_end_index][self.tag2id_onerel['HE-TE']] == 1:
                        break
                    if shaking_tag[r_index][h_start_index][t_end_index][self.tag2id_onerel['HB-TB']] == 1 and t_end_index!= t_start_index:
                        break
                    if shaking_tag[r_index][h_start_index][t_end_index][self.tag2id_onerel['HB-TE']] == 1:
                        for h_end_index in range(h_start_index, seq_lens):
                            if shaking_tag[r_index][h_end_index][t_end_index][self.tag2id_onerel['HE-TE']] ==1:
                                subj_tok_span = [h_start_index,h_end_index+1]
                                obj_tok_span = [t_start_index,t_end_index+1]
                                subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
                                obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
                                subject = text[subj_char_span[0]:subj_char_span[1]]
                                object = text[obj_char_span[0]:obj_char_span[1]]
                                rel_list.append({
                                    "subject": subject,
                                    "object": object,
                                    "subj_tok_span": subj_tok_span,
                                    "obj_tok_span": obj_tok_span,
                                    "subj_char_span": subj_char_span,
                                    "obj_char_span": obj_char_span,
                                    "predicate": self.id2rel[r_index],
                                })
                                break
                        break
            
            # h_end_index = spot[1]
            # t_end_index = spot[2]
            
            # if tag == self.tag2id_onerel['HE-TE']:
            #     for h_start_index in range(h_end_index,0):
            #         if shaking_tag[r_index][h_start_index][t_end_index][self.tag2id_onerel['HB-TE']] == 1:
            #             for t_start_index in range(t_end_index,0):
            #                 if shaking_tag[r_index][h_start_index][t_start_index][self.tag2id_onerel['HB-TB']] ==1:
            #                     subj_tok_span = [h_start_index,h_end_index+1]
            #                     obj_tok_span = [t_start_index,t_end_index+1]
            #                     subj_char_span = [tok2char_span[h_start_index][0],tok2char_span[h_end_index][1]]
            #                     obj_char_span = [tok2char_span[t_start_index][0],tok2char_span[t_end_index][1]]
            #                     subject = text[subj_char_span[0]:subj_char_span[1]]
            #                     object = text[obj_char_span[0]:obj_char_span[1]]
            #                     rel_list.append({
            #                         "subject": subject,
            #                         "object": object,
            #                         "subj_tok_span": subj_tok_span,
            #                         "obj_tok_span": obj_tok_span,
            #                         "subj_char_span": subj_char_span,
            #                         "obj_char_span": obj_char_span,
            #                         "predicate": self.id2rel[r_index],
            #                     })
            #                     break
            #             break


        return rel_list



class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(HandshakingTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], [] 

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0], self.tag2id_head_rel["REL-OH2SH"]))
                
            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id_tail_rel["REL-OT2ST"]))
                
        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return: 
            shake_seq_tag: (rel_size, shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(len(self.rel2id), shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
            shaking_seq_tag[sp[0]][shaking_ind] = sp[3]
        return shaking_seq_tag


    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return: 
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      text, 
                      ent_shaking_tag, 
                      head_rel_shaking_tag, 
                      tail_rel_shaking_tag, 
                      tok2char_span, 
                      tok_offset = 0, char_offset = 0):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        '''
        rel_list = []
        
        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(ent_shaking_tag)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(head_rel_shaking_tag)
        tail_rel_matrix_spots = self.get_spots_fr_shaking_tag(tail_rel_shaking_tag)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2]
            if tag_id != self.tag2id_ent["ENT-H2T"]:
                continue
            
            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]] 
            
            head_key = sp[0] # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append({
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            })
            
        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_tail_rel["REL-ST2OT"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[1], sp[2])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel["REL-OT2ST"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[2], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            
            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                subj_head_key, obj_head_key = sp[1], sp[2]
            elif tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                subj_head_key, obj_head_key = sp[2], sp[1]
                
            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[subj_head_key] # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key] # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = "{}-{}-{}".format(rel_id, subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation 
                        continue
                    
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": self.id2rel[rel_id],
                    })
        return rel_list


class MyEnDecodeScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(MyEnDecodeScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        head_rel_matrix_spots, tail_rel_matrix_spots = [], [], [] 

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0], self.tag2id_head_rel["REL-OH2SH"]))
                
            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id_tail_rel["REL-OT2ST"]))
                
        return head_rel_matrix_spots, tail_rel_matrix_spots
    def get_spots_my(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots_all,ent_rel_matrix_spots, head_rel_matrix_spots = [], [] ,[]

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            ent_matrix_spots_all.append((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots_all.append((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_rel_matrix_spots.append((self.rel2id[rel["predicate"]],subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_rel_matrix_spots.append((self.rel2id[rel["predicate"]],obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0], self.tag2id_head_rel["REL-OH2SH"]))
        return ent_rel_matrix_spots, head_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return: 
            shake_seq_tag: (rel_size, shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(len(self.rel2id), shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
            shaking_seq_tag[sp[0]][shaking_ind] = sp[3]
        return shaking_seq_tag


    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return: 
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                      text, 
                      ent_shaking_tag, 
                      head_rel_shaking_tag,
                      tok2char_span, 
                      tok_offset = 0, char_offset = 0):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        '''
        rel_list = []
        
        ent_matrix_spots = self.get_spots_fr_shaking_tag(ent_shaking_tag)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(head_rel_shaking_tag)

        # entity
        rel_ind_head_ind2entities = {}
        for sp in ent_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id != self.tag2id_ent["ENT-H2T"]:
                continue
            
            char_span_list = tok2char_span[sp[1]:sp[2] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]] 
            
            head_key = sp[1] # take head as the key to entity list start with the head token

            if rel_id not in rel_ind_head_ind2entities:
                rel_ind_head_ind2entities[rel_id] = {}
            if head_key not in rel_ind_head_ind2entities[rel_id]:
                rel_ind_head_ind2entities[rel_id][head_key] = []
            ent_temp = {
                "text": ent_text,
                "tok_span": [sp[1], sp[2] + 1],
                "char_span": char_sp,
            }
            if ent_temp not in rel_ind_head_ind2entities[rel_id][head_key]:
                rel_ind_head_ind2entities[rel_id][head_key].append(ent_temp)


        #head relation
        rel_head_link = {}
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                sub_head = sp[1]
                obj_head = sp[2]
            if tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                sub_head = sp[2]
                obj_head = sp[1]
            if rel_id not in rel_head_link:
                rel_head_link[rel_id] = []
            if (sub_head,obj_head) not in rel_head_link[rel_id]:
                rel_head_link[rel_id].append((sub_head,obj_head))
        

        rel_list = []
        for rel_id in rel_head_link:
            if rel_id not in rel_ind_head_ind2entities:
                continue
            for sh,oh in rel_head_link[rel_id]:
                if sh not in rel_ind_head_ind2entities[rel_id]:
                    continue
                if oh not in rel_ind_head_ind2entities[rel_id]:
                    continue
                for sub_entiy_info in rel_ind_head_ind2entities[rel_id][sh]:
                    for obj_entity_info in rel_ind_head_ind2entities[rel_id][oh]:
                        if sub_entiy_info["text"]!=obj_entity_info["text"]:
                            rel_list.append({
                                "subject": sub_entiy_info["text"],
                                "object": obj_entity_info["text"],
                                "subj_tok_span": [sub_entiy_info["tok_span"][0] + tok_offset, sub_entiy_info["tok_span"][1] + tok_offset],
                                "obj_tok_span": [obj_entity_info["tok_span"][0] + tok_offset, obj_entity_info["tok_span"][1] + tok_offset],
                                "subj_char_span": [sub_entiy_info["char_span"][0] + char_offset, sub_entiy_info["char_span"][1] + char_offset],
                                "obj_char_span": [obj_entity_info["char_span"][0] + char_offset, obj_entity_info["char_span"][1] + char_offset],
                                "predicate": self.id2rel[rel_id],
                            })


        

        return rel_list



class ODRTEDataMaker4Bert():
    def __init__(self, tokenizer, handshaking_tagger,rel_neg_sample_num):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger
        self.rel_num = len(self.handshaking_tagger.id2rel)
        self.rel_neg_sample_num = rel_neg_sample_num
    
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = True, 
                                    truncation = True,
                                    return_special_tokens_mask = True,
                                    padding = 'max_length',max_length=max_seq_len+2)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            sep_index = codes["input_ids"].index(102)
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]
            
            rel_potential_vec = torch.zeros(self.rel_num).long()
            rel_potential_list = []
            for rel_spot in spots_tuple[1]:
                rel_index = rel_spot[0]
                rel_potential_vec[rel_index] = 1
                if rel_index not in rel_potential_list:
                    rel_potential_list.append(rel_index)

            sample_tp = (sample,
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                     rel_potential_vec,
                     rel_potential_list,
                     sep_index
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
 
    def generate_batch_train(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        hb_tb_matrix_spots_list = []
        hb_te_matrix_spots_list = []
        he_te_matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_neg_sam_list=self.get_rel_neg_sam_list(tp[7])
            rel_potential_list_list.append(rel_neg_sam_list)
            sep_index_list.append(tp[8])

            
            hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots = tp[5]
            hb_tb_matrix_spots_list.append( hb_tb_matrix_spots)
            hb_te_matrix_spots_list.append(hb_te_matrix_spots)
            he_te_matrix_spots_list.append(he_te_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        
        batch_shaking_tag,global_region_tag = self.handshaking_tagger.spots2batch_tag(hb_tb_matrix_spots_list, hb_te_matrix_spots_list, he_te_matrix_spots_list)
        if data_type != "test":
            batch_shaking_tag = self.rel_shaking_tag_neg_sample(batch_shaking_tag,rel_potential_list_list)

        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_shaking_tag, batch_rel_potential_vec,rel_potential_list_list,sep_index_list,global_region_tag
    def generate_batch_valid(self, batch_data, data_type = "test"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        hb_tb_matrix_spots_list = []
        hb_te_matrix_spots_list = []
        he_te_matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_neg_sam_list=self.get_rel_neg_sam_list(tp[7])
            rel_potential_list_list.append(rel_neg_sam_list)
            sep_index_list.append(tp[8])

            
            hb_tb_matrix_spots,hb_te_matrix_spots,he_te_matrix_spots = tp[5]
            hb_tb_matrix_spots_list.append( hb_tb_matrix_spots)
            hb_te_matrix_spots_list.append(hb_te_matrix_spots)
            he_te_matrix_spots_list.append(he_te_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        
        batch_shaking_tag,global_region_tag = self.handshaking_tagger.spots2batch_tag(hb_tb_matrix_spots_list, hb_te_matrix_spots_list, he_te_matrix_spots_list)
        if data_type != "test":
            batch_shaking_tag = self.rel_shaking_tag_neg_sample(batch_shaking_tag,rel_potential_list_list)

        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_shaking_tag, batch_rel_potential_vec,rel_potential_list_list,sep_index_list
    
    def get_rel_neg_sam_list(self,rel_potential_list):
        rel_index_list = list(range(self.rel_num))
        pos_rel_num = len(rel_potential_list)
        neg_rel_num = self.rel_neg_sample_num-pos_rel_num
        for pos_rel_index in rel_potential_list:
            rel_index_list.remove(pos_rel_index)
        sam_neg_rel_list  = random.sample(rel_index_list,neg_rel_num)
        sam_rel_index_list = (rel_potential_list+sam_neg_rel_list)
        sam_rel_index_list.sort()
        return torch.tensor(sam_rel_index_list)
    def rel_shaking_tag_neg_sample(self,rel_shaking_tag,rel_potential_list_list):
        negsam_rel_shaking_tag_list = []
        for sam_rel_shaking_tag,rel_neg_sam in  zip(rel_shaking_tag,rel_potential_list_list):
            negsam_rel_shaking_tag_list.append(torch.index_select(sam_rel_shaking_tag,0,rel_neg_sam))
        return torch.stack(negsam_rel_shaking_tag_list,dim=0)
    def get_2d_attention_mask(self,attention_mask,sep_index,max_seq_len=100):
        attention_mask = attention_mask[None,:]+attention_mask[:,None]
        attention_mask[self.rel_num+2:sep_index-1,1:self.rel_num]=0
        attention_mask = attention_mask.ne(0).long()
        return attention_mask

class DataMaker4Bert():
    def __init__(self, tokenizer, handshaking_tagger,rel_neg_sample_num):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger
        self.rel_num = len(self.handshaking_tagger.id2rel)
        self.rel_neg_sample_num = rel_neg_sample_num
    
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = True, 
                                    truncation = True,
                                    return_special_tokens_mask = True,
                                    padding = 'max_length',max_length=max_seq_len+2)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor([102]+codes["input_ids"][1:]).long()
            attention_mask = torch.tensor([1]*(self.rel_num+1)+codes["attention_mask"]).long()
            sep_index = codes["input_ids"].index(102)+self.rel_num+1
            attention_mask = self.get_2d_attention_mask(attention_mask,sep_index,max_seq_len)
            token_type_ids = torch.tensor((self.rel_num+2)*[0]+(len(codes["token_type_ids"])-1)*[1]).long()
            tok2char_span = codes["offset_mapping"][1:-1]
            
            rel_potential_vec = torch.zeros(self.rel_num).long()
            rel_potential_list = []
            for rel_spot in spots_tuple[1]:
                rel_index = rel_spot[0]
                rel_potential_vec[rel_index] = 1
                if rel_index not in rel_potential_list:
                    rel_potential_list.append(rel_index)

            sample_tp = (sample,
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                     rel_potential_vec,
                     rel_potential_list,
                     sep_index
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    def get_indexed_data_only_bert(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = True, 
                                    truncation = True,
                                    return_special_tokens_mask = True,
                                    padding = 'max_length',max_length=max_seq_len+2)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            sep_index = codes["input_ids"].index(102)
            input_ids = torch.tensor(codes["input_ids"]).long()
            input_ids[sep_index] = 0
            input_ids[-1] = 102

            attention_mask = torch.tensor(codes["attention_mask"]).long()
            attention_mask[sep_index] = 0
            attention_mask[-1] = 0

            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"][1:-1]
            
            rel_potential_vec = torch.zeros(self.rel_num).long()
            rel_potential_list = []
            for rel_spot in spots_tuple[1]:
                rel_index = rel_spot[0]
                rel_potential_vec[rel_index] = 1
                if rel_index not in rel_potential_list:
                    rel_potential_list.append(rel_index)

            sample_tp = (sample,
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                     rel_potential_vec,
                     rel_potential_list,
                     sep_index
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
 
    def generate_batch_train(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_neg_sam_list=self.get_rel_neg_sam_list(tp[7])
            rel_potential_list_list.append(rel_neg_sam_list)
            sep_index_list.append(tp[8])

            
            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)
            batch_head_rel_shaking_tag = self.rel_shaking_tag_neg_sample(batch_head_rel_shaking_tag,rel_potential_list_list)
            batch_tail_rel_shaking_tag = self.rel_shaking_tag_neg_sample(batch_tail_rel_shaking_tag,rel_potential_list_list)


        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag, \
                batch_rel_potential_vec,rel_potential_list_list,sep_index_list
    def generate_batch_valid(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_potential_list_list.append(tp[7])
            sep_index_list.append(tp[8])

            
            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)


        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag, \
                batch_rel_potential_vec,rel_potential_list_list,sep_index_list
    def get_rel_neg_sam_list(self,rel_potential_list):
        rel_index_list = list(range(self.rel_num))
        pos_rel_num = len(rel_potential_list)
        neg_rel_num = self.rel_neg_sample_num-pos_rel_num
        for pos_rel_index in rel_potential_list:
            rel_index_list.remove(pos_rel_index)
        sam_neg_rel_list  = random.sample(rel_index_list,neg_rel_num)
        sam_rel_index_list = (rel_potential_list+sam_neg_rel_list)
        sam_rel_index_list.sort()
        return torch.tensor(sam_rel_index_list)
    def rel_shaking_tag_neg_sample(self,rel_shaking_tag,rel_potential_list_list):
        negsam_rel_shaking_tag_list = []
        for sam_rel_shaking_tag,rel_neg_sam in  zip(rel_shaking_tag,rel_potential_list_list):
            negsam_rel_shaking_tag_list.append(torch.index_select(sam_rel_shaking_tag,0,rel_neg_sam))
        return torch.stack(negsam_rel_shaking_tag_list,dim=0)
    def get_2d_attention_mask(self,attention_mask,sep_index,max_seq_len=100):
        attention_mask = attention_mask[None,:]+attention_mask[:,None]
        attention_mask[self.rel_num+2:sep_index-1,1:self.rel_num]=0
        attention_mask = attention_mask.ne(0).long()
        return attention_mask
class MyDataMaker4Bert():
    def __init__(self, tokenizer, handshaking_tagger,rel_neg_sample_num):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger
        self.rel_num = len(self.handshaking_tagger.id2rel)
        self.rel_neg_sample_num = rel_neg_sample_num
    
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = True, 
                                    truncation = True,
                                    return_special_tokens_mask = True,
                                    padding = 'max_length',max_length=max_seq_len+2)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots_my(sample)

            # get codes
            input_ids = torch.tensor([102]+codes["input_ids"][1:]).long()
            attention_mask = torch.tensor([1]*(self.rel_num+1)+codes["attention_mask"]).long()
            token_type_ids = torch.tensor((self.rel_num+2)*[0]+(len(codes["token_type_ids"])-1)*[1]).long()
            tok2char_span = codes["offset_mapping"][1:-1]
            sep_index = codes["input_ids"].index(102)+self.rel_num+1
            rel_potential_vec = torch.zeros(self.rel_num).long()
            rel_potential_list = []
            for rel_spot in spots_tuple[1]:
                rel_index = rel_spot[0]
                rel_potential_vec[rel_index] = 1
                if rel_index not in rel_potential_list:
                    rel_potential_list.append(rel_index)

            sample_tp = (sample,
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                     rel_potential_vec,
                     rel_potential_list,
                     sep_index
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    def get_indexed_data_only_bert(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = True, 
                                    truncation = True,
                                    return_special_tokens_mask = True,
                                    padding = 'max_length',max_length=max_seq_len+2)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            sep_index = codes["input_ids"].index(102)
            input_ids = torch.tensor(codes["input_ids"]).long()
            input_ids[sep_index] = 0
            input_ids[-1] = 102

            attention_mask = torch.tensor(codes["attention_mask"]).long()
            attention_mask[sep_index] = 0
            attention_mask[-1] = 0

            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"][1:-1]
            
            rel_potential_vec = torch.zeros(self.rel_num).long()
            rel_potential_list = []
            for rel_spot in spots_tuple[1]:
                rel_index = rel_spot[0]
                rel_potential_vec[rel_index] = 1
                if rel_index not in rel_potential_list:
                    rel_potential_list.append(rel_index)

            sample_tp = (sample,
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                     rel_potential_vec,
                     rel_potential_list,
                     sep_index
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
 
    def generate_batch_train(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        ent_rel_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_neg_sam_list=self.get_rel_neg_sam_list(tp[7])
            rel_potential_list_list.append(rel_neg_sam_list)
            sep_index_list.append(tp[8])

            
            if data_type != "test":
                ent_rel_matrix_spots, head_rel_matrix_spots = tp[5]
                ent_rel_spots_list.append(ent_rel_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        batch_ent_rel_shaking_tag, batch_head_rel_shaking_tag = None, None
        if data_type != "test":
            batch_ent_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(ent_rel_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_ent_rel_shaking_tag = self.rel_shaking_tag_neg_sample(batch_ent_rel_shaking_tag,rel_potential_list_list)
            batch_head_rel_shaking_tag = self.rel_shaking_tag_neg_sample(batch_head_rel_shaking_tag,rel_potential_list_list)
            


        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_ent_rel_shaking_tag, batch_head_rel_shaking_tag, \
                batch_rel_potential_vec,rel_potential_list_list,sep_index_list
    def generate_batch_valid(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        rel_potential_vec_list = []
        rel_potential_list_list = []
        sep_index_list = []

        
        ent_rel_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            rel_potential_vec_list.append(tp[6])
            rel_potential_list_list.append(tp[7])
            sep_index_list.append(tp[8])

            
            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots = tp[5]
                ent_rel_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        batch_rel_potential_vec = torch.stack(rel_potential_vec_list,dim = 0)
        
        batch_ent_rel_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(ent_rel_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)


        return sample_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_ent_rel_shaking_tag, batch_head_rel_shaking_tag, \
                batch_rel_potential_vec,rel_potential_list_list,sep_index_list
    def get_rel_neg_sam_list(self,rel_potential_list):
        rel_index_list = list(range(self.rel_num))
        pos_rel_num = len(rel_potential_list)
        neg_rel_num = self.rel_neg_sample_num-pos_rel_num
        for pos_rel_index in rel_potential_list:
            rel_index_list.remove(pos_rel_index)
        sam_neg_rel_list  = random.sample(rel_index_list,neg_rel_num)
        sam_rel_index_list = (rel_potential_list+sam_neg_rel_list)
        sam_rel_index_list.sort()
        return torch.tensor(sam_rel_index_list)
    def rel_shaking_tag_neg_sample(self,rel_shaking_tag,rel_potential_list_list):
        negsam_rel_shaking_tag_list = []
        for sam_rel_shaking_tag,rel_neg_sam in  zip(rel_shaking_tag,rel_potential_list_list):
            negsam_rel_shaking_tag_list.append(torch.index_select(sam_rel_shaking_tag,0,rel_neg_sam))
        return torch.stack(negsam_rel_shaking_tag_list,dim=0)
            

class DataMaker4BiLSTM():
    def __init__(self, text2indices, get_tok2char_span_map, handshaking_tagger):
        self.text2indices = text2indices
        self.handshaking_tagger = handshaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map
        
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]

            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample,
                     input_ids,
                     tok2char_span,
                     spots_tuple,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_batch(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])    
            tok2char_span_list.append(tp[2])
            
            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[3]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return sample_list, \
                batch_input_ids, tok2char_span_list, \
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag
    
class TPLinkerRelBert(nn.Module):
    def __init__(self, encoder, 
                 rel_size, 
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist,
                 rel_neg_sample_num,
                 device
                ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.rel_size = rel_size
        self.hidden_size_table = hidden_size
        self.rel_neg_sample_num = rel_neg_sample_num
        self.bert_out_drop = torch.nn.Dropout(0.2)
        self.shaking_hands_drop = torch.nn.Dropout(0.1)
        
        
        self.shaking_type = shaking_type
        self.device = device
        if self.shaking_type == "cat_rel_emb":
            self.rel_pred_fc = nn.Linear(hidden_size,2)
            self.ent_fc = nn.Linear(2*hidden_size, self.hidden_size_table)
            self.ent_fc_out = nn.Linear(self.hidden_size_table, 2)
            self.shaking_hiddens_fc = nn.Linear(2*hidden_size,self.hidden_size_table)
            self.rel_emb_fc = nn.Linear(hidden_size,self.hidden_size_table)
            self.rel_emb_size4tabel = 100
            self.rel_emb_4tabel_fc = nn.Linear(self.hidden_size_table,self.rel_emb_size4tabel)


            bound = 1 / math.sqrt(self.hidden_size_table)
            self.weight_4_head_rel = Parameter(torch.empty(self.rel_size,self.rel_emb_size4tabel+self.hidden_size_table,3))
            self.bias_4_head_rel = Parameter(torch.empty(self.rel_size,3))
            self.weight_4_tail_rel = Parameter(torch.empty(self.rel_size,self.rel_emb_size4tabel+self.hidden_size_table,3))
            self.bias_4_tail_rel = Parameter(torch.empty(self.rel_size,3))

            nn.init.uniform_(self.weight_4_head_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_head_rel, -bound, bound)
            nn.init.uniform_(self.weight_4_tail_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_tail_rel, -bound, bound)

            self.register_parameter("weight_4_head_rel", self.weight_4_head_rel)
            self.register_parameter("bias_4_head_rel", self.bias_4_head_rel)
            self.register_parameter("weight_4_tail_rel", self.weight_4_tail_rel)
            self.register_parameter("bias_4_tail_rel", self.bias_4_tail_rel)

        else:
            self.ent_fc = nn.Linear(hidden_size, 2)
            self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            
            for ind, fc in enumerate(self.head_rel_fc_list):
                self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
            for ind, fc in enumerate(self.tail_rel_fc_list):
                self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)


            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)
        
                # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None # it will be set in the first forwarding
        
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist
        
    def forward(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)

        last_hidden_state = context_outputs[0]

        word_embeds_output,rel_embeds_output = self.get_word_emb(last_hidden_state,sep_index_list)
        rel_embeds_output = torch.tanh(rel_embeds_output)
        rel_pred_outputs = self.rel_pred_fc(rel_embeds_output)
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(word_embeds_output)
        if self.shaking_type != "cat_rel_emb":
            shaking_hiddens4ent = shaking_hiddens
            shaking_hiddens4rel = shaking_hiddens
            
            
            # add distance embeddings if it is set
            if self.dist_emb_size != -1:
                # set self.dist_embbedings
                hidden_size = shaking_hiddens.size()[-1]
                if self.dist_embbedings is None:
                    dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                    for d in range(self.dist_emb_size):
                        for i in range(hidden_size):
                            if i % 2 == 0:
                                dist_emb[d][i] = math.sin(d / 10000**(i / hidden_size))
                            else:
                                dist_emb[d][i] = math.cos(d / 10000**((i - 1) / hidden_size))
                    seq_len = input_ids.size()[1]
                    dist_embbeding_segs = []
                    for after_num in range(seq_len, 0, -1):
                        dist_embbeding_segs.append(dist_emb[:after_num, :])
                    self.dist_embbedings = torch.cat(dist_embbeding_segs, dim = 0)
                
                if self.ent_add_dist:
                    shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                if self.rel_add_dist:
                    shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                    
    #         if self.dist_emb_size != -1 and self.ent_add_dist:
    #             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4ent = shaking_hiddens
    #         if self.dist_emb_size != -1 and self.rel_add_dist:
    #             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4rel = shaking_hiddens
                
            ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
            head_rel_shaking_outputs_list = []
            tail_rel_shaking_outputs_list = []
            for fc in self.head_rel_fc_list:
                head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
            for fc in self.tail_rel_fc_list:
                tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        else:
            if is_Train:
                rel_embeds = self.rel_emb_fc(rel_embeds_output)
                shaking_hiddens = self.shaking_hands_drop(torch.tanh(self.shaking_hiddens_fc(shaking_hiddens)))
                
                ent_shaking_outputs = self.ent_fc_out(shaking_hiddens)

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape
                rel_sample_index_texsor = torch.stack(rel_potential_list_list,dim=0).long()

                #对关系进行动态抽样
                rel_embeds = rel_embeds.gather(dim=1,index = rel_sample_index_texsor[:,:,None].expand(-1,-1,rel_embeds.shape[-1]))

                # shaking_hiddens = torch.tanh(shaking_hiddens[:,None,:,:]+rel_embeds[:,:,None,:])

            
                shaking_hiddens4rel = torch.cat((shaking_hiddens[:,None,:,:].expand(-1,self.rel_neg_sample_num,-1,-1),self.rel_emb_4tabel_fc(rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1))),-1) 
                bingxing = True
                if bingxing:
                    weight_4_head_rel_after_rel_sam = self.weight_4_head_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,rel_embeds.shape[-1]+self.rel_emb_size4tabel,3))
                    weight_4_tail_rel_after_rel_sam = self.weight_4_tail_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,rel_embeds.shape[-1]+self.rel_emb_size4tabel,3))
                    bias_4_head_rel_rel_sam = self.bias_4_head_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,3))
                    bias_4_tail_rel_rel_sam = self.bias_4_tail_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,3))
                    head_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[:,:,None,:]
                    tail_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_tail_rel_after_rel_sam)+bias_4_tail_rel_rel_sam[:,:,None,:]
                else:
                    pass

            else:
                #测试部分代码，使用全部关系进行解码。
                rel_embeds = self.rel_emb_fc(rel_embeds_output)
                shaking_hiddens = torch.tanh(self.shaking_hiddens_fc(shaking_hiddens))
                
                ent_shaking_outputs = self.ent_fc_out(shaking_hiddens)

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape

                # shaking_hiddens = torch.tanh(shaking_hiddens[:,None,:,:]+rel_embeds[:,:,None,:])
            
                shaking_hiddens4rel = torch.cat((shaking_hiddens[:,None,:,:].expand(-1,self.rel_size,-1,-1),self.rel_emb_4tabel_fc(rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1))),-1)

                weight_4_head_rel_after_rel_sam = self.weight_4_head_rel[None,:,:,:].expand(batch_size,-1,-1,-1)
                weight_4_tail_rel_after_rel_sam = self.weight_4_tail_rel[None,:,:,:].expand(batch_size,-1,-1,-1)
                bias_4_head_rel_rel_sam = self.bias_4_head_rel[None,:,:].expand(batch_size,-1,-1)
                bias_4_tail_rel_rel_sam = self.bias_4_tail_rel[None,:,:].expand(batch_size,-1,-1)
                head_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[:,:,None,:]
                tail_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_tail_rel_after_rel_sam)+bias_4_tail_rel_rel_sam[:,:,None,:]
        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs,rel_pred_outputs
    def get_word_emb(self,last_hidden_state,sep_index_list):
        word_hidden_state_list = []
        rel_hidden_state_list = []
        for batch_i,batch_h in enumerate(last_hidden_state):
            wordsplice1 = batch_h[(self.rel_size)+2:sep_index_list[batch_i]]
            wordsplice2 = batch_h[sep_index_list[batch_i]+1:]
            rel_emb_temp = batch_h[1:(self.rel_size)+1]
            word_emb_temp = torch.cat((wordsplice1,wordsplice2),dim=0)
            rel_hidden_state_list.append(rel_emb_temp)
            word_hidden_state_list.append(word_emb_temp)
        return torch.stack(word_hidden_state_list,dim=0),torch.stack(rel_hidden_state_list,dim=0)
    def get_rel_shaking_outputs_after_neg_sam(self,rel_embeds_output,shaking_hiddens4rel,rel_potential_list_list,rel_type = "head"):
        if rel_type == "head":
            fc_list = self.head_rel_fc_list
        else:
            fc_list = self.tail_rel_fc_list
        batch,shaking_seq_len,dim = shaking_hiddens4rel.shape
        rel_shaking_outputs_list = []
        for batch_ind, rel_potential_list in enumerate(rel_potential_list_list):
            shaking_hiddens4rel_batch_ind = shaking_hiddens4rel[batch_ind]
            rel_shaking_outputs_list_batch_ind = []
            for sam_rel_index in rel_potential_list:
                sam_rel_index = sam_rel_index.item()
                rel_embeds_batch_sam = rel_embeds_output[batch_ind][sam_rel_index][None,:].repeat(shaking_seq_len,1)
                fc = fc_list[sam_rel_index]
                rel_shaking_outputs_list_batch_ind.append(torch.tanh(fc(torch.cat([shaking_hiddens4rel_batch_ind,rel_embeds_batch_sam],dim=-1))))
            rel_shaking_outputs_list_batch_ind = torch.stack(rel_shaking_outputs_list_batch_ind,dim=0)
            rel_shaking_outputs_list.append(rel_shaking_outputs_list_batch_ind)
        rel_shaking_outputs_list = torch.stack(rel_shaking_outputs_list,dim=0)
        return rel_shaking_outputs_list
    def get_rel_emb_after_neg_sam(self,rel_embeds_output,rel_potential_list_list):
        rel_embeds_output_neg_sam_list = []
        for rel_embeds_output_batch_i,rel_potential_batch_i in zip(rel_embeds_output,rel_potential_list_list):
            rel_embeds_output_batch_i_neg_sam = rel_embeds_output_batch_i.index_select(dim = 0,index = rel_potential_batch_i)
            rel_embeds_output_neg_sam_list.append(rel_embeds_output_batch_i_neg_sam)
        rel_embeds_output_neg_sam = torch.stack(rel_embeds_output_neg_sam_list,dim=0)
        return rel_embeds_output_neg_sam

class MyRelBert(nn.Module):
    #实体表二分类，关系表三分类。共两个表
    def __init__(self, encoder, 
                 rel_size, 
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist,
                 rel_neg_sample_num,
                 device
                ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.rel_size = rel_size
        self.hidden_size_table = hidden_size
        self.rel_neg_sample_num = rel_neg_sample_num
        self.bert_out_drop = torch.nn.Dropout(0.2)
        self.shaking_hands_drop = torch.nn.Dropout(0.1)
        
        
        self.shaking_type = shaking_type
        self.device = device
        if self.shaking_type == "cat_rel_emb":
            self.rel_pred_fc = nn.Linear(hidden_size,2)
            self.ent_fc = nn.Linear(2*hidden_size, self.hidden_size_table)
            self.ent_fc_out = nn.Linear(self.hidden_size_table, 2)
            self.shaking_hiddens_fc = nn.Linear(2*hidden_size,self.hidden_size_table)
            self.rel_emb_fc = nn.Linear(hidden_size,self.hidden_size_table)
            self.rel_emb_size4tabel = 100
            self.rel_emb_4tabel_fc = nn.Linear(self.hidden_size_table,self.rel_emb_size4tabel)


            bound = 1 / math.sqrt(self.hidden_size_table)
            self.weight_4_head_rel = Parameter(torch.empty(self.rel_size,self.hidden_size_table,3))
            self.bias_4_head_rel = Parameter(torch.empty(self.rel_size,3))
            self.weight_4_ent_rel = Parameter(torch.empty(self.rel_size,self.hidden_size_table,1))
            self.bias_4_ent_rel = Parameter(torch.empty(self.rel_size,1))

            nn.init.uniform_(self.weight_4_head_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_head_rel, -bound, bound)
            nn.init.uniform_(self.weight_4_ent_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_ent_rel, -bound, bound)

            self.register_parameter("weight_4_head_rel", self.weight_4_head_rel)
            self.register_parameter("bias_4_head_rel", self.bias_4_head_rel)
            self.register_parameter("weight_4_ent_rel", self.weight_4_ent_rel)
            self.register_parameter("bias_4_ent_rel", self.bias_4_ent_rel)

        else:
            self.ent_fc = nn.Linear(hidden_size, 2)
            self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            
            for ind, fc in enumerate(self.head_rel_fc_list):
                self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
            for ind, fc in enumerate(self.tail_rel_fc_list):
                self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)


            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)
        
                # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None # it will be set in the first forwarding
        
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist
        
    def forward(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)

        last_hidden_state = context_outputs[0]

        word_embeds_output,rel_embeds_output = self.get_word_emb(last_hidden_state,sep_index_list)
        rel_embeds_output = torch.tanh(rel_embeds_output)
        rel_pred_outputs = self.rel_pred_fc(rel_embeds_output)
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(word_embeds_output)
        if self.shaking_type != "cat_rel_emb":
            shaking_hiddens4ent = shaking_hiddens
            shaking_hiddens4rel = shaking_hiddens
            
            
            # add distance embeddings if it is set
            if self.dist_emb_size != -1:
                # set self.dist_embbedings
                hidden_size = shaking_hiddens.size()[-1]
                if self.dist_embbedings is None:
                    dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                    for d in range(self.dist_emb_size):
                        for i in range(hidden_size):
                            if i % 2 == 0:
                                dist_emb[d][i] = math.sin(d / 10000**(i / hidden_size))
                            else:
                                dist_emb[d][i] = math.cos(d / 10000**((i - 1) / hidden_size))
                    seq_len = input_ids.size()[1]
                    dist_embbeding_segs = []
                    for after_num in range(seq_len, 0, -1):
                        dist_embbeding_segs.append(dist_emb[:after_num, :])
                    self.dist_embbedings = torch.cat(dist_embbeding_segs, dim = 0)
                
                if self.ent_add_dist:
                    shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                if self.rel_add_dist:
                    shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                    
    #         if self.dist_emb_size != -1 and self.ent_add_dist:
    #             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4ent = shaking_hiddens
    #         if self.dist_emb_size != -1 and self.rel_add_dist:
    #             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4rel = shaking_hiddens
                
            ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
            head_rel_shaking_outputs_list = []
            tail_rel_shaking_outputs_list = []
            for fc in self.head_rel_fc_list:
                head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
            for fc in self.tail_rel_fc_list:
                tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        else:
            if is_Train:
                rel_embeds = self.rel_emb_fc(rel_embeds_output)
                shaking_hiddens = self.shaking_hands_drop(torch.tanh(self.shaking_hiddens_fc(shaking_hiddens)))

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape
                rel_sample_index_texsor = torch.stack(rel_potential_list_list,dim=0).long()

                #对关系进行动态抽样
                rel_embeds = rel_embeds.gather(dim=1,index = rel_sample_index_texsor[:,:,None].expand(-1,-1,rel_embeds.shape[-1]))

                # shaking_hiddens = torch.tanh(shaking_hiddens[:,None,:,:]+rel_embeds[:,:,None,:])

            
                #shaking_hiddens4rel = torch.cat((shaking_hiddens[:,None,:,:].expand(-1,self.rel_neg_sample_num,-1,-1),self.rel_emb_4tabel_fc(rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1))),-1) 
                shaking_hiddens4rel = shaking_hiddens[:,None,:,:].expand(-1,self.rel_neg_sample_num,-1,-1)+rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1)
                #shaking_hiddens4rel = shaking_hiddens[:,None,:,:].expand(-1,self.rel_neg_sample_num,-1,-1)
                bingxing = True
                if bingxing:
                    weight_4_head_rel_after_rel_sam = self.weight_4_head_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,rel_embeds.shape[-1],3))
                    weight_4_ent_rel_after_rel_sam = self.weight_4_ent_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,rel_embeds.shape[-1],1))
                    bias_4_head_rel_rel_sam = self.bias_4_head_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,3))
                    bias_4_ent_rel_rel_sam = self.bias_4_ent_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,1))
                    head_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[:,:,None,:]
                    ent_rel_shaking_outputs = torch.sigmoid(torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_ent_rel_after_rel_sam)+bias_4_ent_rel_rel_sam[:,:,None,:])
                else:
                    pass

            else:
                #测试部分代码，使用全部关系进行解码。
                rel_embeds = self.rel_emb_fc(rel_embeds_output)
                shaking_hiddens = torch.tanh(self.shaking_hiddens_fc(shaking_hiddens))

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape

                # shaking_hiddens = torch.tanh(shaking_hiddens[:,None,:,:]+rel_embeds[:,:,None,:])
            
                #shaking_hiddens4rel = torch.cat((shaking_hiddens[:,None,:,:].expand(-1,self.rel_size,-1,-1),self.rel_emb_4tabel_fc(rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1))),-1)
                shaking_hiddens4rel = shaking_hiddens[:,None,:,:].expand(-1,self.rel_size,-1,-1)+rel_embeds[:,:,None,:].expand(-1,-1,shaking_seq_len,-1)
                weight_4_head_rel_after_rel_sam = self.weight_4_head_rel[None,:,:,:].expand(batch_size,-1,-1,-1)
                weight_4_ent_rel_after_rel_sam = self.weight_4_ent_rel[None,:,:,:].expand(batch_size,-1,-1,-1)
                bias_4_head_rel_rel_sam = self.bias_4_head_rel[None,:,:].expand(batch_size,-1,-1)
                bias_4_ent_rel_rel_sam = self.bias_4_ent_rel[None,:,:].expand(batch_size,-1,-1)
                head_rel_shaking_outputs = torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[:,:,None,:]
                ent_rel_shaking_outputs = torch.sigmoid(torch.einsum('abcd,abde->abce',shaking_hiddens4rel,weight_4_ent_rel_after_rel_sam)+bias_4_ent_rel_rel_sam[:,:,None,:])
        return ent_rel_shaking_outputs.squeeze(-1), head_rel_shaking_outputs,rel_pred_outputs
    def get_word_emb(self,last_hidden_state,sep_index_list):
        word_hidden_state_list = []
        rel_hidden_state_list = []
        for batch_i,batch_h in enumerate(last_hidden_state):
            wordsplice1 = batch_h[(self.rel_size)+2:sep_index_list[batch_i]]
            wordsplice2 = batch_h[sep_index_list[batch_i]+1:]
            rel_emb_temp = batch_h[1:(self.rel_size)+1]
            word_emb_temp = torch.cat((wordsplice1,wordsplice2),dim=0)
            rel_hidden_state_list.append(rel_emb_temp)
            word_hidden_state_list.append(word_emb_temp)
        return torch.stack(word_hidden_state_list,dim=0),torch.stack(rel_hidden_state_list,dim=0)
    def get_rel_shaking_outputs_after_neg_sam(self,rel_embeds_output,shaking_hiddens4rel,rel_potential_list_list,rel_type = "head"):
        if rel_type == "head":
            fc_list = self.head_rel_fc_list
        else:
            fc_list = self.tail_rel_fc_list
        batch,shaking_seq_len,dim = shaking_hiddens4rel.shape
        rel_shaking_outputs_list = []
        for batch_ind, rel_potential_list in enumerate(rel_potential_list_list):
            shaking_hiddens4rel_batch_ind = shaking_hiddens4rel[batch_ind]
            rel_shaking_outputs_list_batch_ind = []
            for sam_rel_index in rel_potential_list:
                sam_rel_index = sam_rel_index.item()
                rel_embeds_batch_sam = rel_embeds_output[batch_ind][sam_rel_index][None,:].repeat(shaking_seq_len,1)
                fc = fc_list[sam_rel_index]
                rel_shaking_outputs_list_batch_ind.append(torch.tanh(fc(torch.cat([shaking_hiddens4rel_batch_ind,rel_embeds_batch_sam],dim=-1))))
            rel_shaking_outputs_list_batch_ind = torch.stack(rel_shaking_outputs_list_batch_ind,dim=0)
            rel_shaking_outputs_list.append(rel_shaking_outputs_list_batch_ind)
        rel_shaking_outputs_list = torch.stack(rel_shaking_outputs_list,dim=0)
        return rel_shaking_outputs_list
    def get_rel_emb_after_neg_sam(self,rel_embeds_output,rel_potential_list_list):
        rel_embeds_output_neg_sam_list = []
        for rel_embeds_output_batch_i,rel_potential_batch_i in zip(rel_embeds_output,rel_potential_list_list):
            rel_embeds_output_batch_i_neg_sam = rel_embeds_output_batch_i.index_select(dim = 0,index = rel_potential_batch_i)
            rel_embeds_output_neg_sam_list.append(rel_embeds_output_batch_i_neg_sam)
        rel_embeds_output_neg_sam = torch.stack(rel_embeds_output_neg_sam_list,dim=0)
        return rel_embeds_output_neg_sam

class TPLinkerBert(nn.Module):
    def __init__(self, encoder, 
                 rel_size, 
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist,
                 rel_neg_sample_num,
                 device
                ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.rel_size = rel_size
        # self.hidden_size_table = int(hidden_size/3)
        self.hidden_size_table = hidden_size
        self.rel_neg_sample_num = rel_neg_sample_num
        
        
        self.shaking_type = shaking_type
        self.device = device
        if self.shaking_type == "cat_rel_emb":
            self.rel_pred_fc = nn.Linear(hidden_size,2)
            self.ent_fc = nn.Linear(2*hidden_size, self.hidden_size_table)
            self.ent_fc_out = nn.Linear(self.hidden_size_table, 2)
            self.shaking_hiddens_fc = nn.Linear(2*hidden_size,self.hidden_size_table)
            self.rel_emb_fc = nn.Linear(hidden_size,self.hidden_size_table)


            bound = 1 / math.sqrt(self.hidden_size_table)
            self.weight_4_head_rel = Parameter(torch.empty(self.rel_size,self.hidden_size_table,3))
            self.bias_4_head_rel = Parameter(torch.empty(self.rel_size,3))
            self.weight_4_tail_rel = Parameter(torch.empty(self.rel_size,self.hidden_size_table,3))
            self.bias_4_tail_rel = Parameter(torch.empty(self.rel_size,3))

            nn.init.uniform_(self.weight_4_head_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_head_rel, -bound, bound)
            nn.init.uniform_(self.weight_4_tail_rel, -bound, bound)
            nn.init.uniform_(self.bias_4_tail_rel, -bound, bound)

            self.register_parameter("weight_4_head_rel", self.weight_4_head_rel)
            self.register_parameter("bias_4_head_rel", self.bias_4_head_rel)
            self.register_parameter("weight_4_tail_rel", self.weight_4_tail_rel)
            self.register_parameter("bias_4_tail_rel", self.bias_4_tail_rel)
            
            # self.head_rel_fc_list = [nn.Linear(self.hidden_size_table, 3) for _ in range(rel_size)]
            # self.tail_rel_fc_list = [nn.Linear(self.hidden_size_table, 3) for _ in range(rel_size)]
            
            # for ind, fc in enumerate(self.head_rel_fc_list):
            #     self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            #     self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
            # for ind, fc in enumerate(self.tail_rel_fc_list):
            #     self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            #     self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        else:
            self.ent_fc = nn.Linear(hidden_size, 2)
            self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
            
            for ind, fc in enumerate(self.head_rel_fc_list):
                self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
            for ind, fc in enumerate(self.tail_rel_fc_list):
                self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
                self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)


            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)
        
                # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None # it will be set in the first forwarding
        
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist
        
    def forward(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        #word_embeds_output = self.get_word_emb(last_hidden_state,sep_index_list)
        word_embeds_output = last_hidden_state[:,1:-1,:]
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(word_embeds_output)
        if self.shaking_type != "cat_rel_emb":
            shaking_hiddens4ent = shaking_hiddens
            shaking_hiddens4rel = shaking_hiddens
            
            
            # add distance embeddings if it is set
            if self.dist_emb_size != -1:
                # set self.dist_embbedings
                hidden_size = shaking_hiddens.size()[-1]
                if self.dist_embbedings is None:
                    dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                    for d in range(self.dist_emb_size):
                        for i in range(hidden_size):
                            if i % 2 == 0:
                                dist_emb[d][i] = math.sin(d / 10000**(i / hidden_size))
                            else:
                                dist_emb[d][i] = math.cos(d / 10000**((i - 1) / hidden_size))
                    seq_len = input_ids.size()[1]
                    dist_embbeding_segs = []
                    for after_num in range(seq_len, 0, -1):
                        dist_embbeding_segs.append(dist_emb[:after_num, :])
                    self.dist_embbedings = torch.cat(dist_embbeding_segs, dim = 0)
                
                if self.ent_add_dist:
                    shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                if self.rel_add_dist:
                    shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                    
    #         if self.dist_emb_size != -1 and self.ent_add_dist:
    #             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4ent = shaking_hiddens
    #         if self.dist_emb_size != -1 and self.rel_add_dist:
    #             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
    #         else:
    #             shaking_hiddens4rel = shaking_hiddens
                
            ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
            head_rel_shaking_outputs_list = []
            tail_rel_shaking_outputs_list = []
            for fc in self.head_rel_fc_list:
                head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
            for fc in self.tail_rel_fc_list:
                tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        else:
            if is_Train:
                shaking_hiddens = torch.tanh(self.shaking_hiddens_fc(shaking_hiddens))
                
                ent_shaking_outputs = self.ent_fc_out(shaking_hiddens)

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape
                rel_sample_index_texsor = torch.stack(rel_potential_list_list,dim=0).long()

                #对关系进行动态抽样
                # rel_embeds = rel_embeds.gather(dim=1,index = rel_sample_index_texsor[:,:,None].expand(-1,-1,rel_embeds.shape[-1]))

                # shaking_hiddens = torch.tanh(shaking_hiddens[:,None,:,:]+rel_embeds[:,:,None,:])
                #shaking_hiddens = shaking_hiddens4ent[:,None,:,:].expand(-1,self.rel_neg_sample_num,-1,-1)

            
                shaking_hiddens4rel = shaking_hiddens
                if self.rel_neg_sample_num < self.rel_size:
                    weight_4_head_rel_after_rel_sam = self.weight_4_head_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,self.hidden_size_table,3))
                    weight_4_tail_rel_after_rel_sam = self.weight_4_tail_rel[None,:,:,:].expand(batch_size,-1,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None,None].expand(-1,-1,self.hidden_size_table,3))
                    bias_4_head_rel_rel_sam = self.bias_4_head_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,3))
                    bias_4_tail_rel_rel_sam = self.bias_4_tail_rel[None,:,:].expand(batch_size,-1,-1).gather(dim=1,index=rel_sample_index_texsor[:,:,None].expand(-1,-1,3))
                    head_rel_shaking_outputs = torch.einsum('acd,abde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[:,:,None,:]
                    tail_rel_shaking_outputs = torch.einsum('acd,abde->abce',shaking_hiddens4rel,weight_4_tail_rel_after_rel_sam)+bias_4_tail_rel_rel_sam[:,:,None,:]
                else:
                    weight_4_head_rel_after_rel_sam = self.weight_4_head_rel
                    weight_4_tail_rel_after_rel_sam = self.weight_4_tail_rel
                    bias_4_head_rel_rel_sam = self.bias_4_head_rel
                    bias_4_tail_rel_rel_sam = self.bias_4_tail_rel
                    head_rel_shaking_outputs = torch.einsum('acd,bde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam) + bias_4_head_rel_rel_sam[None,:,None,:]
                    tail_rel_shaking_outputs = torch.einsum('acd,bde->abce',shaking_hiddens4rel,weight_4_tail_rel_after_rel_sam) + bias_4_tail_rel_rel_sam[None,:,None,:]


            else:
                #测试部分代码，使用全部关系进行解码。
                shaking_hiddens = torch.tanh(self.shaking_hiddens_fc(shaking_hiddens))
                
                ent_shaking_outputs = self.ent_fc_out(shaking_hiddens)

                batch_size,shaking_seq_len,_ = shaking_hiddens.shape
            
                shaking_hiddens4rel = shaking_hiddens

                weight_4_head_rel_after_rel_sam = self.weight_4_head_rel
                weight_4_tail_rel_after_rel_sam = self.weight_4_tail_rel
                bias_4_head_rel_rel_sam = self.bias_4_head_rel
                bias_4_tail_rel_rel_sam = self.bias_4_tail_rel
                head_rel_shaking_outputs = torch.einsum('acd,bde->abce',shaking_hiddens4rel,weight_4_head_rel_after_rel_sam)+bias_4_head_rel_rel_sam[None,:,None,:]
                tail_rel_shaking_outputs = torch.einsum('acd,bde->abce',shaking_hiddens4rel,weight_4_tail_rel_after_rel_sam)+bias_4_tail_rel_rel_sam[None,:,None,:]
        rel_pred_outputs = rel_potential_vec
        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs,rel_pred_outputs
    def get_word_emb(self,last_hidden_state,sep_index_list):
        word_hidden_state_list = []
        for batch_i,batch_h in enumerate(last_hidden_state):
            wordsplice1 = batch_h[1:sep_index_list[batch_i]]
            wordsplice2 = batch_h[sep_index_list[batch_i]+1:]
            word_emb_temp = torch.cat((wordsplice1,wordsplice2),dim=0)
            word_hidden_state_list.append(word_emb_temp)
        return torch.stack(word_hidden_state_list,dim=0)
    def get_rel_shaking_outputs_after_neg_sam(self,rel_embeds_output,shaking_hiddens4rel,rel_potential_list_list,rel_type = "head"):
        if rel_type == "head":
            fc_list = self.head_rel_fc_list
        else:
            fc_list = self.tail_rel_fc_list
        batch,shaking_seq_len,dim = shaking_hiddens4rel.shape
        rel_shaking_outputs_list = []
        for batch_ind, rel_potential_list in enumerate(rel_potential_list_list):
            shaking_hiddens4rel_batch_ind = shaking_hiddens4rel[batch_ind]
            rel_shaking_outputs_list_batch_ind = []
            for sam_rel_index in rel_potential_list:
                sam_rel_index = sam_rel_index.item()
                rel_embeds_batch_sam = rel_embeds_output[batch_ind][sam_rel_index][None,:].repeat(shaking_seq_len,1)
                fc = fc_list[sam_rel_index]
                rel_shaking_outputs_list_batch_ind.append(torch.tanh(fc(torch.cat([shaking_hiddens4rel_batch_ind,rel_embeds_batch_sam],dim=-1))))
            rel_shaking_outputs_list_batch_ind = torch.stack(rel_shaking_outputs_list_batch_ind,dim=0)
            rel_shaking_outputs_list.append(rel_shaking_outputs_list_batch_ind)
        rel_shaking_outputs_list = torch.stack(rel_shaking_outputs_list,dim=0)
        return rel_shaking_outputs_list
    def get_rel_emb_after_neg_sam(self,rel_embeds_output,rel_potential_list_list):
        rel_embeds_output_neg_sam_list = []
        for rel_embeds_output_batch_i,rel_potential_batch_i in zip(rel_embeds_output,rel_potential_list_list):
            rel_embeds_output_batch_i_neg_sam = rel_embeds_output_batch_i.index_select(dim = 0,index = rel_potential_batch_i)
            rel_embeds_output_neg_sam_list.append(rel_embeds_output_batch_i_neg_sam)
        rel_embeds_output_neg_sam = torch.stack(rel_embeds_output_neg_sam_list,dim=0)
        return rel_embeds_output_neg_sam



                





class TPLinkerBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix, 
                 emb_dropout_rate, 
                 enc_hidden_size, 
                 dec_hidden_size, 
                 rnn_dropout_rate,
                 rel_size, 
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size, 
                 ent_add_dist, 
                 rel_add_dist):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1], 
                        enc_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.dec_lstm = nn.LSTM(enc_hidden_size, 
                        dec_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        
        hidden_size = dec_hidden_size
           
        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        
        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)
        
        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None # it will be set in the first forwarding
        
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens
        
        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000**(i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000**((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim = 0)
            
            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                
            
        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
        
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
            
        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
        
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim = 1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim = 1)
        
        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class MetricsCalculator_ODRTE():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
            
    def get_rel_cpg(self, sample_list, tok2char_span_list, 
                 batch_pred_taging_outputs, 
                 pattern = "only_head_text"):

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_taging_outputs = batch_pred_taging_outputs[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_taging_outputs,  
                                                      tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])


            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1

class MetricsCalculator_onerel():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
            
    def get_rel_cpg(self, sample_list, tok2char_span_list, 
                 batch_pred_taging_outputs, 
                 pattern = "only_head_text"):

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_taging_outputs = batch_pred_taging_outputs[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_taging_outputs,  
                                                      tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])


            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1
                # else:
                #     pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, pred_taging_outputs, tok2char_span)

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1

class MetricsCalculator_tplinker():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
        
    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc
    
    def get_rel_cpg(self, sample_list, tok2char_span_list, 
                 batch_pred_ent_shaking_outputs,
                 batch_pred_head_rel_shaking_outputs,
                 batch_pred_tail_rel_shaking_outputs, 
                 pattern = "only_head_text"):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim = -1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim = -1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim = -1)
        # batch_pred_ent_shaking_tag = batch_pred_ent_shaking_outputs
        # batch_pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_outputs
        # batch_pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_outputs
        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_ent_shaking_tag, 
                                                      pred_head_rel_shaking_tag, 
                                                      pred_tail_rel_shaking_tag, 
                                                      tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])


            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1
    
class MetricsCalculator():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
        
    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc
    def get_sample_accuracy_entity(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = pred
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc
    
    def get_rel_cpg(self, sample_list, tok2char_span_list, 
                 batch_pred_ent_shaking_outputs,
                 batch_pred_head_rel_shaking_outputs, 
                 pattern = "only_head_text"):
        batch_pred_ent_shaking_tag = batch_pred_ent_shaking_outputs>0.5
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim = -1)

        # batch_pred_ent_shaking_tag = batch_pred_ent_shaking_outputs
        # batch_pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_outputs

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_ent_shaking_tag, 
                                                      pred_head_rel_shaking_tag, 
                                                      tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])


            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1 

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1



class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """

        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        self.weights = self.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        #self.register_buffer('weights', weights)
        #self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb


class RelModel(nn.Module):
    def __init__(self, bert,rel_size,tag_size,predict_threshold):
        super(RelModel, self).__init__()
        self.bert_dim = 768
        self.bert_encoder = bert
        self.rel_size = rel_size
        self.tag_size = tag_size
        self.relation_matrix = nn.Linear(3*self.bert_dim, self.rel_size * self.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, 3*self.bert_dim)
        self.predict_threshold = predict_threshold

        self.dropout = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.activation = nn.ReLU()
    def forward(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        context_outputs = self.bert_encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        encoded_text = context_outputs[0]
        encoded_text = self.dropout(encoded_text)
        #shaking hand
        batch_size, seq_len, bert_dim = encoded_text.size()
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        entity_pairs = self.projection_matrix(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.rel_size, self.tag_size).permute(0,3,1,2,4)


        if is_Train:
            rel_potential_tensor = torch.stack(rel_potential_list_list,dim=0)[:,:,None,None,None].expand(-1,-1,seq_len,seq_len,self.tag_size)
            triple_scores = triple_scores.gather(dim=1,index = rel_potential_tensor)
            return triple_scores
        else:
            return (torch.sigmoid(triple_scores)>self.predict_threshold).long()

class RelModel_OD_RTE(nn.Module):
    def __init__(self, bert,rel_size,tag_size,max_seq_len,device):
        super(RelModel_OD_RTE, self).__init__()
        self.bert_dim = 768
        self.device = device
        self.bert_encoder = bert
        self.rel_size = rel_size
        self.tag_size = tag_size
        self.relation_matrix = nn.Linear(3*self.bert_dim, self.rel_size * self.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, 3*self.bert_dim)
        self.global_region_matrix = nn.Linear(3*self.bert_dim,1)
        self.table_pos_bias = nn.Parameter(torch.rand(256, 768))
        #self.pos_emb_w = RelativeSinusoidalPositionalEmbedding(3*self.bert_dim,0,2*max_seq_len+10)
        #self.pos_embeding = nn.Embedding.from_pretrained(self.pos_emb_w.weights,freeze=True)

        self.dropout = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.activation = nn.ReLU()
    def forward(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        context_outputs = self.bert_encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        encoded_text = context_outputs[0]
        encoded_text = self.dropout(encoded_text)
        #shaking hand
        batch_size, seq_len, bert_dim = encoded_text.size()
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        #tail_representation = encoded_text.repeat(1, seq_len, 1)
        tail_representation = self.table_pos_bias[0:seq_len,:].unsqueeze(0).repeat(batch_size,1,1).repeat(1, seq_len, 1)
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        entity_pairs = self.projection_matrix(entity_pairs)

        # pos_emb = self.get_pos_emb(seq_len)
        # pos_emb = pos_emb[None,:,:].expand(batch_size,-1,-1)

        # entity_pairs = entity_pairs + pos_emb
        
        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.rel_size, self.tag_size).permute(0,3,1,2,4)


        if is_Train:
            rel_potential_tensor = torch.stack(rel_potential_list_list,dim=0)[:,:,None,None,None].expand(-1,-1,seq_len,seq_len,self.tag_size)
            triple_scores = triple_scores.gather(dim=1,index = rel_potential_tensor)
            global_region_scores = self.global_region_matrix(entity_pairs)
            return triple_scores,global_region_scores
        else:
            return torch.sigmoid(triple_scores)
    def forward2(self, input_ids, attention_mask, token_type_ids, rel_potential_vec,rel_potential_list_list,sep_index_list,is_Train = True):
        context_outputs = self.bert_encoder(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        encoded_text = context_outputs[0]
        encoded_text = self.dropout(encoded_text)
        #shaking hand
        batch_size, seq_len, bert_dim = encoded_text.size()
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        entity_pairs = self.projection_matrix(entity_pairs)

        # pos_emb = self.get_pos_emb(seq_len)
        # pos_emb = pos_emb[None,:,:].expand(batch_size,-1,-1)

        # entity_pairs = entity_pairs + pos_emb
        
        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.rel_size, self.tag_size).permute(0,3,1,2,4)


        if is_Train:
            rel_potential_tensor = torch.stack(rel_potential_list_list,dim=0)[:,:,None,None,None].expand(-1,-1,seq_len,seq_len,self.tag_size)
            triple_scores = triple_scores.gather(dim=1,index = rel_potential_tensor)
            global_region_scores = self.global_region_matrix(entity_pairs)
            return triple_scores,global_region_scores
        else:
            return torch.sigmoid(triple_scores)
    def get_pos_emb(self,seq_len):
        i_index = torch.arange(0,seq_len)[None,:]
        j_index = torch.arange(0,seq_len)[:,None]
        Rij = (i_index-j_index+seq_len+5).long()
        pos_emb = self.pos_embeding(Rij.to(self.device)).reshape(seq_len*seq_len,-1)

        return pos_emb