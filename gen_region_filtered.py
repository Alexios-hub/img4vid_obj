import torch
from torchvision import models, transforms
import stanza
from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.data.path.append('./models/nltk_data')
from nltk.corpus import wordnet as wn
import json

from PIL import Image

from src.utils.tsv_file import TSVFile
import os
from src.datasets.data_utils.image_ops import img_from_base64
from tqdm import tqdm
import gc



resnet = models.resnet101(pretrained=True).to(torch.device('cuda'))
resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False


nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos",
            dir="./models/stanfordnlp/stanza/",
            download_method=None,
        )

word2vec = KeyedVectors.load_word2vec_format(
            "./models/word2vec/GoogleNews-vectors-negative300.bin.gz", binary=True
        )


def get_region_filtered(frames_cap_str,video_id,raw_frames):
    # 处理文本
    doc = nlp(frames_cap_str)
    # 提取名词
    nouns = [
        word.text
        for sent in doc.sentences
        for word in sent.words
        if word.upos == "NOUN"
    ]
    nouns_num_dic = dict()
    for noun in nouns:
        nouns_num_dic.setdefault(noun, 0)
        nouns_num_dic[noun] += 1
    nouns_unit = list(nouns_num_dic.keys())  # 去除重复单词

    ##过滤出wordnet中属于object的单词
    def is_word_under_object(word):
        # 查找word的所有同义词集
        synsets = wn.synsets(word)

        # 对于每个同义词集
        for synset in synsets:
            # 获取该同义词集到'entity.n.01'（最顶层实体）的hypernym_paths
            hypernym_paths = synset.hypernym_paths()

            # 检查每条路径
            for path in hypernym_paths:
                # 检查路径中是否有'object'这个上位词
                for hypernym in path:
                    if hypernym.name().split(".")[0] == "object":
                        return True
        return False
    
    nouns_unit = [noun for noun in nouns_unit if is_word_under_object(noun)]#筛选出物体，可能为空数组
    dbscan = DBSCAN(eps=0.5, min_samples=1, metric="precomputed")
    if len(nouns_unit) == 0:
        res={}
        for i in range(1,33):
            res[str(i)]=[]
        return res

    nouns_unit_vec = np.array([word2vec[word] for word in nouns_unit if word in word2vec])
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(nouns_unit_vec)
    similarity_matrix = np.clip(similarity_matrix, -1, 1)
    # 将相似度转换为距离（DBSCAN需要距离矩阵）
    distance_matrix = 1 - similarity_matrix
    # 应用DBSCAN聚类
    # eps参数设置为0.5，表示如果两个点的距离小于0.5（即相似度大于0.5），则它们属于同一个聚类
    # min_samples设置为1，因为我们不需要考虑最小样本数的限制
    clusters = dbscan.fit_predict(distance_matrix)
    # 初始化聚类计数字典，用于存储每个聚类的样本数量
    cluster_counts = {}

    # 遍历每个单词及其对应的聚类标签
    for word, cluster_label in zip(nouns_unit, clusters):
        # 如果当前聚类标签还未在字典中，则初始化计数为0
        if cluster_label not in cluster_counts:
            cluster_counts[cluster_label] = 0
        # 累加当前单词的数量到对应的聚类计数上
        cluster_counts[cluster_label] += nouns_num_dic[word]
    # 筛选出样本数量前10的聚类
    cluster_counts_list = sorted(
        cluster_counts.items(), key=lambda x: x[1], reverse=True
    )
    top_clusters = [item[0] for item in cluster_counts_list[:10]]
    filtered_nouns_unit = []

    # 遍历每个名词及其对应的聚类标签
    for word, cluster_label in zip(nouns_unit, clusters):
        if cluster_label in top_clusters:
            filtered_nouns_unit.append(word)
    # 遍历每个名词向量及其对应的聚类标签
    filtered_nouns_unit_vec = []
    for word_vec, cluster_label in zip(nouns_unit_vec, clusters):
        if cluster_label in top_clusters:
            filtered_nouns_unit_vec.append(word_vec)
    
    # 获取faster_rcnn输出结果
    obj_file_path = (
        "datasets/MSRVTT-v2/objects/32frames/faster_rcnn/"
        + video_id
        + ".json"
    )
    with open(obj_file_path, "r") as f:
        obj_dic = json.load(f)

    def is_in_top_clusters(label, filtered_nouns_unit_vec, eps=0.5):
        labels = label.split(" ")#可能有两个
        for l in labels:
            if l in word2vec:
                l_vec = word2vec[l].reshape(1, -1)
                # 计算新单词与已有单词列表中每个单词的余弦相似度
                sim = cosine_similarity(l_vec, filtered_nouns_unit_vec)
                dis = 1 - sim
                min_dis = np.min(dis)
                if min_dis < eps:
                    return True
        return False

    def filter_obj_dic(obj_dic, filtered_nouns_unit_vec, eps=0.5):
        obj_dic_filtered = dict()
        for key in obj_dic.keys():
            frame_obj_arr = []
            if obj_dic[key] != None:
                for obj in obj_dic[key]:
                    if is_in_top_clusters(obj["cls"], filtered_nouns_unit_vec, eps):
                        frame_obj_arr.append(obj)
            obj_dic_filtered[key] = frame_obj_arr
        return obj_dic_filtered

    obj_dic_filtered = filter_obj_dic(obj_dic, filtered_nouns_unit_vec, dbscan.eps)
    return obj_dic_filtered

def get_region_feats(frames_cap_str,video_id,raw_frames):
    # 处理文本
    doc = nlp(frames_cap_str)
    # 提取名词
    nouns = [
        word.text
        for sent in doc.sentences
        for word in sent.words
        if word.upos == "NOUN"
    ]
    nouns_num_dic = dict()
    for noun in nouns:
        nouns_num_dic.setdefault(noun, 0)
        nouns_num_dic[noun] += 1
    nouns_unit = list(nouns_num_dic.keys())  # 去除重复单词

    ##过滤出wordnet中属于object的单词
    def is_word_under_object(word):
        # 查找word的所有同义词集
        synsets = wn.synsets(word)

        # 对于每个同义词集
        for synset in synsets:
            # 获取该同义词集到'entity.n.01'（最顶层实体）的hypernym_paths
            hypernym_paths = synset.hypernym_paths()

            # 检查每条路径
            for path in hypernym_paths:
                # 检查路径中是否有'object'这个上位词
                for hypernym in path:
                    if hypernym.name().split(".")[0] == "object":
                        return True
        return False
    
    nouns_unit = [noun for noun in nouns_unit if is_word_under_object(noun)]
    nouns_unit_vec = np.array([word2vec[word] for word in nouns_unit if word in word2vec])
    if len(nouns_unit) == 0:
        res={}
        for i in range(1,33):
            res[str(i)]=[]
        return res
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(nouns_unit_vec)
    similarity_matrix = np.clip(similarity_matrix, -1, 1)
    # 将相似度转换为距离（DBSCAN需要距离矩阵）
    distance_matrix = 1 - similarity_matrix
    # 应用DBSCAN聚类
    # eps参数设置为0.5，表示如果两个点的距离小于0.5（即相似度大于0.5），则它们属于同一个聚类
    # min_samples设置为1，因为我们不需要考虑最小样本数的限制
    dbscan = DBSCAN(eps=0.5, min_samples=1, metric="precomputed")
    clusters = dbscan.fit_predict(distance_matrix)
    # 初始化聚类计数字典，用于存储每个聚类的样本数量
    cluster_counts = {}

    # 遍历每个单词及其对应的聚类标签
    for word, cluster_label in zip(nouns_unit, clusters):
        # 如果当前聚类标签还未在字典中，则初始化计数为0
        if cluster_label not in cluster_counts:
            cluster_counts[cluster_label] = 0
        # 累加当前单词的数量到对应的聚类计数上
        cluster_counts[cluster_label] += nouns_num_dic[word]
    # 筛选出样本数量前10的聚类
    cluster_counts_list = sorted(
        cluster_counts.items(), key=lambda x: x[1], reverse=True
    )
    top_clusters = [item[0] for item in cluster_counts_list[:10]]
    filtered_nouns_unit = []

    # 遍历每个名词及其对应的聚类标签
    for word, cluster_label in zip(nouns_unit, clusters):
        if cluster_label in top_clusters:
            filtered_nouns_unit.append(word)
    # 遍历每个名词向量及其对应的聚类标签
    filtered_nouns_unit_vec = []
    for word_vec, cluster_label in zip(nouns_unit_vec, clusters):
        if cluster_label in top_clusters:
            filtered_nouns_unit_vec.append(word_vec)
    
    # 获取faster_rcnn输出结果
    obj_file_path = (
        "datasets/MSRVTT-v2/objects/32frames/faster_rcnn/"
        + video_id
        + ".json"
    )
    with open(obj_file_path, "r") as f:
        obj_dic = json.load(f)

    def is_in_top_clusters(label, filtered_nouns_unit_vec, eps=0.5):
        labels = label.split(" ")#可能有两个
        for l in labels:
            if l in word2vec:
                l_vec = word2vec[l].reshape(1, -1)
                # 计算新单词与已有单词列表中每个单词的余弦相似度
                sim = cosine_similarity(l_vec, filtered_nouns_unit_vec)
                dis = 1 - sim
                min_dis = np.min(dis)
                if min_dis < eps:
                    return True
        return False#遍历结束返回False

    def filter_obj_dic(obj_dic, filtered_nouns_unit_vec, eps=0.5):
        obj_dic_filtered = dict()
        for key in obj_dic.keys():
            frame_obj_arr = []
            if obj_dic[key] != None:
                for obj in obj_dic[key]:
                    if is_in_top_clusters(obj["cls"], filtered_nouns_unit_vec, eps):
                        frame_obj_arr.append(obj)
            obj_dic_filtered[key] = frame_obj_arr
        return obj_dic_filtered

    obj_dic_filtered = filter_obj_dic(obj_dic, filtered_nouns_unit_vec, dbscan.eps)
    resnet.eval()
    # 定义图像预处理步骤
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    frame_features = {}
    device = torch.device("cuda")
    
    for frame_idx, bboxes in obj_dic_filtered.items():
        frame_feature_list = []
        for bbox in bboxes:
            try:
                # 裁剪图像区域
                xmin, ymin, xmax, ymax = bbox["bbox"]

                frame = raw_frames[int(frame_idx) - 1]
                frame = torch.permute(frame, (1, 2, 0))
                frame_image = Image.fromarray(
                    frame.numpy().astype("uint8"), "RGB"
                ).crop((xmin, ymin, xmax, ymax))

                # 图像预处理并提取特征
                input_tensor = preprocess(frame_image)
                input_batch = input_tensor.unsqueeze(0)  # 创建一个batch
                input_batch = input_batch.to(device)
                with torch.no_grad():
                    features = resnet(input_batch)
                    # 展平特征张量
                    features = features.view(features.size(0), -1)
                # 特征与bbox特征拼接
                bbox_features = torch.tensor(bbox["bbox"], dtype=torch.float16).to(
                    device
                )
                combined_features = torch.cat((features.squeeze(0), bbox_features), 0)

                frame_feature_list.append(combined_features.half().cpu().detach().numpy().tolist())
            except Exception:
                continue
        frame_features[frame_idx] = frame_feature_list
    return frame_features

#根据视频名称获取帧描述，按索引有序排列
def get_frames_cap(row, frame_cap_dir='datasets/MSRVTT-v2/LLAVA_cap/32frames_cap'):
    """
    parameters:
        row: get_visual_data函数中定义的row
    returns:
        result: 包含每帧caption的数组,按帧递增排列，如果不存在的对应索引存储为""
    """
    result = []
    video_name = os.path.basename(row[0]).split('.')[0]
    for i in range(1,len(row)-1):
        if i<10:
            cap_path = frame_cap_dir + '/' + video_name + '_frame000' + str(i) + '.txt'
        else:
            cap_path = frame_cap_dir + '/' + video_name + '_frame00' + str(i) + '.txt'
        if os.path.exists(cap_path):
            with open(cap_path,'r') as f:
                temp_cap = f.read()
            result.append(temp_cap)
        else:
            result.append("")
    return result

def get_image(bytestring): 
    # output numpy array (T, C, H, W), channel is RGB, T = 1
    cv2_im = img_from_base64(bytestring)
    cv2_im = cv2_im[:,:,::-1] # COLOR_BGR2RGB
    # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    output = np.transpose(cv2_im[np.newaxis, ...], (0, 3, 1, 2))
    return output
    
def get_frames_from_tsv(binary_frms):
    # get pre-extracted video frames from tsv files
    frames = []
    _C, _H, _W = 3, 224, 224

    def sampling(start,end,n):
        if n == 1:
            return [int(round((start+end)/2.))]
        if n < 1:
            raise Exception("behaviour not defined for n<2")
        step = (end-start)/float(n-1)
        return [int(round(start+x*step)) for x in range(n)]

    for i in sampling(0, len(binary_frms)-1, 32):
        try:
            image = get_image(binary_frms[i])#shape=(1, 3, 256, 341)
        except Exception as e:
            print(f"Corrupt frame at {i}")
            image = np.zeros((1, _C, _H, _W), dtype=np.int64)
        _, _C, _H, _W = image.shape
        frames.append(image)
    return np.vstack(frames)

    

def get_visual_data(idx, tsv_file):
    row = tsv_file[idx]
    frames_cap = get_frames_cap(row)
    raw_frames = get_frames_from_tsv(row[2:])
    video_name = os.path.basename(row[0]).split('.')[0]
    return frames_cap,video_name,raw_frames





if __name__ == '__main__':
    #train:0-6512
    #val:0-496(7009)
    #test:0-2989(9999)
    # tsv_path = 'datasets/MSRVTT-v2/frame_tsv/test_32frames_img_size256.img.tsv'
    # tsv_file = TSVFile(tsv_path)
    # frames_cap,video_name,raw_frames = get_visual_data(0, tsv_file)
    # print(tsv_file[0])
    save_dir = 'datasets/MSRVTT-v2/objects/32frames/filtered/'
    #train数据集
    # tsv_path = 'datasets/MSRVTT-v2/frame_tsv/train_32frames_img_size256.img.tsv'
    # tsv_file = TSVFile(tsv_path)
    # for i in tqdm(range(0,6513),desc='Processing training video regions'):
    #     frames_cap,video_name,raw_frames = get_visual_data(i, tsv_file)
    #     frames_cap_str = ''
    #     raw_frames = torch.from_numpy(raw_frames)
    #     for j in range(0,len(frames_cap)):
    #         frames_cap_str += ' ' + frames_cap[j]
    #     frame_region_filtered = get_region_filtered(frames_cap_str=frames_cap_str,video_id=video_name,raw_frames=raw_frames)
    #     json_file = save_dir + video_name + '.json'
    #     with open(json_file,'w') as f:
    #         json.dump(frame_region_filtered,f,indent=4)
    #     del frames_cap, video_name, raw_frames, frame_region_filtered
    #     gc.collect()
    
    # #val数据集
    # tsv_path = 'datasets/MSRVTT-v2/frame_tsv/val_32frames_img_size256.img.tsv'
    # tsv_file = TSVFile(tsv_path)
    # for i in tqdm(range(0,497),desc='Processing val video regions'):
    #     frames_cap,video_name,raw_frames = get_visual_data(i, tsv_file)
    #     frames_cap_str = ''
    #     raw_frames = torch.from_numpy(raw_frames)
    #     for j in range(0,len(frames_cap)):
    #         frames_cap_str += ' ' + frames_cap[j]
    #     frame_region_filtered = get_region_filtered(frames_cap_str=frames_cap_str,video_id=video_name,raw_frames=raw_frames)
    #     json_file = save_dir + video_name + '.json'
    #     with open(json_file,'w') as f:
    #         json.dump(frame_region_filtered,f,indent=4)
    #     del frames_cap, video_name, raw_frames, frame_region_filtered
    #     gc.collect()
    
    #test数据集
    tsv_path = 'datasets/MSRVTT-v2/frame_tsv/test_32frames_img_size256.img.tsv'
    tsv_file = TSVFile(tsv_path)
    for i in tqdm(range(2412,2990),desc='Processing val video regions'):
        frames_cap,video_name,raw_frames = get_visual_data(i, tsv_file)
        frames_cap_str = ''
        raw_frames = torch.from_numpy(raw_frames)
        for j in range(0,len(frames_cap)):
            frames_cap_str += ' ' + frames_cap[j]
        frame_region_filtered = get_region_filtered(frames_cap_str=frames_cap_str,video_id=video_name,raw_frames=raw_frames)
        json_file = save_dir + video_name + '.json'
        with open(json_file,'w') as f:
            json.dump(frame_region_filtered,f,indent=4)
        del frames_cap, video_name, raw_frames, frame_region_filtered
        gc.collect()

