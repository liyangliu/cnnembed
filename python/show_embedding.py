import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize

# img_root = '/home/liyangliu/hdd/ILSVRC2012/resize_256x256/val/img/'
# xy_root = '/home/liyangliu/hdd/ILSVRC2012/resize_256x256/val/xyab/'
img_root = '/home/liyangliu/hdd/ILSVRC2012/resize_256x256/train/img/'
xy_root = '/home/liyangliu/hdd/ILSVRC2012/resize_256x256/train/xyab/'
# seg_root = '/home/liyangliu/hdd/ILSVRC2012/resize_256x256/val/seg/'
# val_list = open('/home/liyangliu/research/color/data/filelist/val_list/val.txt')
val_list = open('/home/liyangliu/research/color/data/filelist/train_list/train_sample_rand.txt')
# vl = val_list.readlines()[:1000]
vl = val_list.readlines()
val_list.close()

feat_name = 'feat_16_sample'
x = np.fromfile('/home/liyangliu/research/bh_tsne/data/%s_tsne.dat'%feat_name, dtype=np.float32).reshape(-1, 2)
x_max = np.max(x)
x_min = np.min(x)
x = (x - x_min) / (x_max - x_min)

S = 5120
G = np.zeros((S, S, 3), np.uint8)
s = 64
sp_num = 256
img_num = 6500
N = img_num*sp_num
cnt = 0
segs = np.fromfile('/home/liyangliu/research/bh_tsne/data/seg_sample_mat.dat', dtype=np.int32).reshape(img_num, sp_num)

for i in range(N):
    # img_idx = i // sp_num
    # img_name = vl[img_idx]
    # patch_idx = i % sp_num

    # img = plt.imread(img_root + img_name[:-1])
    # xyab = np.fromfile(xy_root + img_name[:-5] + 'dat', dtype=np.float64)
    # xs = xyab[:sp_num]
    # ys = xyab[sp_num:sp_num*2]
    # a = np.ceil(x[i,0]*(S-s))
    # b = np.ceil(x[i,1]*(S-s))
    # a = int(a - a%s)
    # b = int(b - b%s)
    # if G[a,b,0] != 0:
        # continue

    # xp = np.round(xs[patch_idx]).astype(np.uint8)
    # yp = np.round(ys[patch_idx]).astype(np.uint8)
    # if len(img.shape) == 2:
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)
    # img[xp-7:xp+7, yp-7:yp+7, :] = np.array([0, 255, 0], dtype=np.uint8)
    # I = imresize(img, [s, s]);
    # G[a:a+s, b:b+s, :] = I

    img_idx = i // sp_num
    img_name = vl[img_idx]
    seg = segs[img_idx]
    patch_idx = i % sp_num

    if seg[patch_idx] == 1:
        img = plt.imread(img_root + img_name[:-1])
        xyab = np.fromfile(xy_root + img_name[:-5] + 'dat', dtype=np.float64)
        xs = xyab[:sp_num]
        ys = xyab[sp_num:sp_num*2]
        a = np.ceil(x[cnt,0]*(S-s))
        b = np.ceil(x[cnt,1]*(S-s))
        cnt += 1
        a = int(a - a%s)
        b = int(b - b%s)
        if G[a,b,0] != 0:
            continue

        xp = np.round(xs[patch_idx]).astype(np.uint8)
        yp = np.round(ys[patch_idx]).astype(np.uint8)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        img[xp-7:xp+7, yp-7:yp+7, :] = np.array([0, 255, 0], dtype=np.uint8)
        I = imresize(img, [s, s]);
        G[a:a+s, b:b+s, :] = I

    # img_idx = i
    # img_name = vl[img_idx]
    # img = plt.imread(img_root + img_name[:-1])

    # if len(img.shape) == 2:
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)

    # I = imresize(img, [s, s]);
    # a = np.ceil(x[i,0]*(S-s))
    # b = np.ceil(x[i,1]*(S-s))
    # a = int(a - a%s)
    # b = int(b - b%s)
    # G[a:a+s, b:b+s, :] = I

    if i%100 == 0:
        print('%d/%d'%(i, N));
plt.imsave('../pics/embed.jpg', G)

# used = []
# qq = len(range(0, S, s))
# abes = np.zeros((qq**2, 2), dtype=np.int32)
# i = 0
# for a in range(0, S, s):
    # for b in range(0, S, s):
        # abes[i] = np.array([a, b], dtype=np.int32)
        # i += 1

# for i in range(abes.shape[0]):
    # a = abes[i, 0]
    # b = abes[i, 1]
    # xf = a / (S - s)
    # yf = b / (S - s)
    # dd = np.sum((x - np.array([xf, yf])) ** 2, 1)
    # dd[used] = np.inf
    # dv = np.min(dd)
    # di = np.argmin(dd)
    # used.append(di)

    # img_idx = di // 16
    # img_name = vl[img_idx]
    # img = plt.imread(img_root + img_name[:-1])
    # xyab = np.fromfile(xy_root + img_name[:-5] + 'dat', dtype=np.float64)
    # xs = xyab[:256]
    # ys = xyab[256:256*2]
    # patch_idx = di % 16
    # xp = np.round(xs[patch_idx]).astype(np.uint8)
    # yp = np.round(ys[patch_idx]).astype(np.uint8)
    # if len(img.shape) == 2:
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)
    # img[xp-7:xp+7, yp-7:yp+7, :] = np.array([0, 255, 0], dtype=np.uint8)
    # I = imresize(img, [s, s]);
    # G[a:a+s, b:b+s, :] = I
    # if i%100 == 0:
        # print('%d/%d'%(i, abes.shape[0]));
# plt.imsave('../pics/embed_nn.jpg', G)
