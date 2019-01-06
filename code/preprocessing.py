import re

def clean_caption(description):
    description=description.lower()
    description=re.sub(r'\b\w\b', '', description)
    description=re.sub(r'\s+', ' ', description)
    description=re.sub('\d+', '',description)
    description=re.sub(r'[^a-zA-Z ]', '',description)
    description=" ".join(description.split())
    return description

def preprocessing():
    file_captions = open('../data/Flickr8k_text/Flickr8k.token.txt','r')
    token=file_captions.read().split('\n')
    file_captions.close()
    captions = {}
    for i in range(len(token)-1):
        id_capt = token[i].split("\t")
        id_capt[0] = id_capt[0][:len(id_capt[0])-2]
        if id_capt[0] not in captions:
            captions[id_capt[0]]=list()
        captions[id_capt[0]].append("startcap "+clean_caption(id_capt[1])+" endcap")
        
    file_token_captions = open("../result/tokenImages.txt",'w')
    for tk_key, tk_value_list in captions.items():
        for tk_value in tk_value_list:
            file_token_captions.write(str(tk_key)+'\t'+str(tk_value)+'\n')
            file_token_captions.flush()
    file_token_captions.close()
        
    file_train_id = open('../data/Flickr8k_text/Flickr_8k.trainImages.txt','r')
    train_id=file_train_id.read().split('\n')[:-1]
    train_captions=dict()
    for key, value in captions.items():
        if key in train_id:
            if key not in train_captions:
                train_captions[key]=value
    
    file_train_captions = open("../result/trainImages.txt",'w')
    for tr_key, tr_value_list in train_captions.items():
        for tr_value in tr_value_list:
            file_train_captions.write(str(tr_key)+'\t'+str(tr_value)+'\n')
            file_train_captions.flush()
    file_train_captions.close()
    
    file_dev_id = open('../data/Flickr8k_text/Flickr_8k.devImages.txt','r')
    dev_id=file_dev_id.read().split('\n')[:-1]
    dev_captions=dict()
    for key, value in captions.items():
        if key in dev_id:
            if key not in dev_captions:
                dev_captions[key]=value
                       
    file_dev_captions = open("../result/devImages.txt",'w')
    for dv_key, dv_value_list in dev_captions.items():
        for dv_value in dv_value_list:
            file_dev_captions.write(str(dv_key)+'\t'+str(dv_value)+'\n')
            file_dev_captions.flush()
    file_dev_captions.close()

    file_test_id = open('../data/Flickr8k_text/Flickr_8k.testImages.txt','r')
    test_id=file_test_id.read().split('\n')[:-1]
    test_captions=dict()
    for key, value in captions.items():
        if key in test_id:
            if key not in test_captions:
                test_captions[key]=value
                       
    file_test_captions = open("../result/testImages.txt",'w')
    for ts_key, ts_value_list in test_captions.items():
        for ts_value in ts_value_list:
            file_test_captions.write(str(ts_key)+'\t'+str(ts_value)+'\n')
            file_test_captions.flush()
    file_test_captions.close()
    
preprocessing()
