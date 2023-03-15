import cv2 as cv
from pathlib import Path
from src import enums
from hloc import extract_features,match_features,extractors,matchers,utils
import torch

# Initialization function for feature extractor and matcher. 
# It should just get the names of chosen extractor and matcher, provided in enum.
def init(used_extractor,used_matcher):
    if used_extractor == enums.Extractors.ORB:
        extractor=cv.ORB_create(nfeatures=2500)
    elif used_extractor == enums.Extractors.SuperPoint:
        feature_conf = extract_features.confs['superpoint_aachen']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = utils.base_model.dynamic_load(extractors, feature_conf['model']['name'])
        extractor = Model(feature_conf['model']).eval().to(device) 
        
    if used_matcher == enums.Matchers.OrbHamming:
        matcher=cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    elif used_matcher == enums.Matchers.SuperGlue:
        matcher_conf = match_features.confs['superglue']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = utils.base_model.dynamic_load(matchers, matcher_conf['model']['name'])
        matcher = Model(matcher_conf['model']).eval().to(device)
    return extractor,matcher


# This function should be called when detecting features.
# img_pth corresponds to the path of the image
# used_matcher is a Enum defied in the slam class that represents which detector should be used
# save is boolean and indicates if the features should be saved as an image on the disk
# out_pth and name correspond to the saved image output pth and name

def detector(img_pth, img_name,extractor,used_extractor=enums.Extractors.ORB, save=False, out_pth=Path(''), name='orb_out.jpg'):
    if used_extractor == enums.Extractors.ORB:
        kp,des=orb_detector(extractor,img_pth/img_name, save, out_pth, name)
        detector = {
            "name": img_name,
            "kp": kp,
            "des": des
        }
        return kp,detector
    elif used_extractor == enums.Extractors.SuperPoint:
        return SuperPoint_detector(extractor,img_pth/img_name,save,out_pth,name)



# This function should be called when matching features.
# img1 corresponds to the first image. It is a dict with attribute names
#     "des"  the descriptor of the features
#     "kp"   the keypoints as pixel indices corresponding to the descriptors
#     "name" the file name of the image
# used_matcher is a Enum defied in the slam class that represents which detector should be used
# save is boolean and indicates if the matches should be saved as an image on the disk
# img_pth corresponds to the path of the image

def matcher(img1, img2, matcher,used_matcher=enums.Matchers.OrbHamming, save=False, img_pth=Path(''), out_pth=Path('')):
    if used_matcher == enums.Matchers.OrbHamming:
        return orb_matcher(matcher,img1, img2, save, img_pth, out_pth)
    elif used_matcher == enums.Matchers.SuperGlue:
        return SuperGlue_matcher(matcher,img1,img2)

# Function called automatically by extractor if ORB is chosen, extracts and uses ORB features
def orb_detector(orb,img_pth,save=False, out_pth=Path(''), name='orb_out.jpg'):
    img = cv.imread(str(img_pth), 0)
    kp, des = orb.detectAndCompute(img, None)
    if save:
        img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv.imwrite(str(out_pth / 'images/detector' / name), img2)
    return kp, des

# Function called automatically by extractor if Superpoint is chosen, extracts and uses SuperPoint features 
@torch.no_grad()
def SuperPoint_detector(extractor,img_pth,save,out_pth,name):
    img = cv.imread(str(img_pth),0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tens=torch.unsqueeze(torch.FloatTensor(img.astype(float)/255),dim=0)
    img_tens=torch.unsqueeze(img_tens,dim=0)
    size_tens=torch.unsqueeze(torch.tensor(img.shape),dim=0)
    data={ 
        'name': [name],
        'image': img_tens,
        'original_size':size_tens,
    }
    pred = extractor(utils.tools.map_tensor(data, lambda x: x.to(device)))
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    torch.squeeze(size_tens,dim=0)
    pred.update({'image_size': size_tens})
    kps=[]
    for kp in pred['keypoints']: 
        kps.append(cv.KeyPoint(kp[0],kp[1],size=1))
    if save:
        img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv.imwrite(str(out_pth / 'images/detector' / name), img2)
    return kps,pred

# Function called automatically by matcher if orb matcher is used, matches ORB features
def orb_matcher(bf,keypoint, query, save=False, img_pth=Path(''), out_pth=Path('')):
    des1 = keypoint["des"]
    des2 = query["des"]
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if save:
        draw_matches(keypoint, query, matches, img_pth, out_pth)
    return matches

# Function called automatically by matcher if SuperGlue matcher is used, matches SuperPoint features
@torch.no_grad()
def SuperGlue_matcher(matcher,img1_data,img2_data):
    data={}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for k, v in img1_data.items():
        data[k+'0'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1,)+(img1_data['image_size'][0][0],img1_data['image_size'][0][1])[::-1])
    for k, v in img2_data.items():
        data[k+'1'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
        data['image1'] = torch.empty((1,)+(img2_data['image_size'][0][0],img2_data['image_size'][0][1])[::-1])
    data = {k: v[None] for k, v in data.items()}
    matches_data = matcher(data)
    matches = matches_data['matches0'][0].cpu().short().numpy()
    scores = matches_data['matching_scores0'][0].cpu().half().numpy()
    matches_list=[]
    i=0
    for match,score in zip(matches,scores):
        if not(match==-1):
            matches_list.append(cv.DMatch(_queryIdx=i,_trainIdx=match,_distance=-score))
        i=i+1
    return sorted(matches_list, key = lambda x:x.distance)

# Visualization function used to draw matches on the images
def draw_matches(current_keyframe, _detector, _matches, img_pth, out_pth):
    img1 = cv.imread(str(img_pth / current_keyframe["name"]), 0)
    img2 = cv.imread(str(img_pth / _detector["name"]), 0)
    # Draw first n matches.
    img3 = cv.drawMatches(img1, current_keyframe["kp"], img2, _detector["kp"], _matches[:10], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    name = "usedMatch_" + current_keyframe["name"] + "_" + _detector["name"] + ".jpg"
    result = cv.imwrite(str(out_pth / 'images/matcher' / name), img3)

# Visualization function used to draw matches on the images
def draw_matches_knn(current_keyframe, _detector, _matches, matchesMask, img_pth, out_pth, indx=0):
    img1 = cv.imread(str(img_pth / current_keyframe["name"]), 0)
    img2 = cv.imread(str(img_pth / _detector["name"]), 0)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, current_keyframe["kp"], img2, _detector["kp"], _matches, None, **draw_params)
    name = "usedMatch_" + str(indx) + ".jpg"
    cv.imwrite(str(out_pth / 'images/matcher' / name), img3)
