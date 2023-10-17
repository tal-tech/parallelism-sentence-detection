import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'module/text_classification'))
from module.text_classification.inference import TextClassifier


config = {
       'checkpoint_lst':[os.path.join(root, 'model/rhetoric_model/Parallel_PretrainedBert_1e-05_16.pt')],
        'use_bert': True,
        'embd_path': os.path.join(root, 'model/word_embedding/tencent_small'),
        'model_config_lst': [{
            'is_state': False,
            'model_name': 'bert',
            'pretrained_model_path': os.path.join(root, 'model/bert_chinese_wwm_ext_pytorch')
        }]
}


if __name__ == "__main__":
    sent_list = [
        '她的眼睛明亮清澈；她的笑容甜美动人；她的声音柔和温暖；她的气质高雅优美，无人能及。'
    ]
    pretrained_model_path = config['model_config_lst'][0]['pretrained_model_path']
    model = TextClassifier(config['embd_path'], config['checkpoint_lst'], config['model_config_lst'],
                           pretrained_model_path)
    max_seq_len =config['max_seq_len'] if 'max_seq_len' in config else 80
    need_mask = config['need_mask'] if 'need_mask' in config else False
    pred_list, proba_list = model.predict_all_mask(sent_list, max_seq_len=max_seq_len, max_batch_size=20,
                                                   need_mask=need_mask)
    pos_sent_list = [sent_list[i] for i in range(len(pred_list)) if pred_list[i] == 1]
    print(pred_list, proba_list)
    print("排比数目", len(pos_sent_list))
    for sent in pos_sent_list:
        print(sent)

