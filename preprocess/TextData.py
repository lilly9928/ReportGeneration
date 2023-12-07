import re
import pandas as pd
import numpy as np

def make_vocab(input_dir,output_dir):
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    caption_length = []
    captions = pd.read_csv(input_dir,header=0)
    set_caption_length = [None] * len(captions)
    all_tag=captions['Problems'].values
    all_caption = captions['findings'].values
    all_impression = captions['impression'].values

    for i in range(len(captions)):
        vocab_set.update(all_tag[i].split(';'))
        sentence=str(all_caption[i]).split('\n')
        for j in range(len(sentence)):
            words = SENTENCE_SPLIT_REGEX.split(sentence[j])
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0,'<pad>')
    vocab_list.insert(1, '<unk>')

    with open(output_dir+'/vocab_caption.txt','w') as f:
        f.writelines([w + '\n' for w in vocab_list])

    print('Make vocabulary for caption')
    print('The number of total words of caption: %d' % len(vocab_set))


def make_txt_file(output_dir,image_info_dir,caption_info_dir):
    caption_info = open(caption_info_dir,'r')
    read_caption_info = pd.read_csv(caption_info)
    read_caption_info=read_caption_info.sort_values(by=['uid'])
    read_caption_info = read_caption_info.reset_index(drop=True)
    caption_index = 0


    image_info = open(image_info_dir,'r')
    read_image_info = pd.read_csv(image_info)
    read_image_info = read_image_info.sort_values(by=['uid'])
    read_image_info = read_image_info.reset_index(drop=True)
    train_range = int(len(read_image_info)*1)


    f = open(output_dir + '/caption.txt', 'w')
    f.writelines('oimage,image,caption,label\n')

    for i in range(0,train_range):

        case_uid = read_image_info['uid'][i]
        caption_uid = read_caption_info['uid'][caption_index]

        if case_uid == caption_uid:
            oimage_name = str(read_image_info['filename'][i].split('.')[0]) + '.dcm.png'
            image_name = str(read_image_info['filename'][i].split('.')[0]) + '.dcm.png.jpg'
            caption = str(read_caption_info['findings'][caption_index]).replace('\n', ' ')
            caption = caption.replace(',', ' ')
            caption = caption.replace('nan', ' ')
            label = str(read_caption_info['Problems'][caption_index]).replace(';', ' ')
            label = label.replace(',', ' ')
            label = label.replace('nan', ' ')
            f.write(oimage_name+','+image_name + ',' + caption + ',' + label + '\n')

        else:
            caption_index+=1

    f.close()

    # f = open(output_dir + '/caption_test.txt', 'w')
    # f.writelines('oimage,image,caption,label\n')
    #
    # for i in range(train_range,len(read_image_info)):
    #
    #     case_uid = read_image_info['uid'][i]
    #     caption_uid = read_caption_info['uid'][caption_index]
    #
    #     if case_uid == caption_uid:
    #         oimage_name = str(read_image_info['filename'][i].split('.')[0]) + '.dcm.png'
    #         image_name = str(read_image_info['filename'][i].split('.')[0])+'.dcm.png.jpg'
    #         caption = str(read_caption_info['findings'][caption_index]).replace('\n',' ')
    #         caption = caption.replace(',',' ')
    #         caption = caption.replace('nan', ' ')
    #         label = str(read_caption_info['Problems'][caption_index]).replace(';',' ')
    #         label = label.replace(',',' ')
    #         label = label.replace('nan', ' ')
    #         f.write(oimage_name + ',' + image_name + ',' + caption + ',' + label + '\n')
    #     else:
    #         caption_index+=1
    #
    # f.close()


if __name__ == '__main__':
    input_dir = 'D:/data/iuct/indiana_reports.csv'
    image_dir = 'D:/data/iuct/indiana_projections.csv'
    output_dir ='D:/data/iuct'
    #make_vocab(input_dir,output_dir)
    make_txt_file(output_dir,image_dir,input_dir)