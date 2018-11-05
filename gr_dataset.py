## Graphical Rectangle Dataset with
import torch
from torch.utils.data import Dataset

import logging as lg
import numpy as np
import cv2 as cv
import random
from utils import *

# ADJACENCY = torch.FloatTensor(
#   [0, 1, 1, 0,
#    1, 0, 0, 1,
#    1, 0, 0, 1,
#    0, 1, 1, 0]
# ).reshape((-1, 4))

ADJACENCY = torch.FloatTensor(
  [0, 1, 1, 0,
   1, 0, 0, 1,
   1, 0, 0, 1,
   0, 1, 1, 0]
).reshape((-1, 4))


def rand_verts() :
  result, _ = torch.rand(2, 2, dtype=torch.float).sort(dim=0)
  result = result.repeat((1, 2)).reshape(-1, 2)
  result[1:3, 1] = torch.from_numpy(
    np.flip(result[1:3, 1]).copy()
  )

  return result

def mtol(adjacency) :
  return [
    tuple(torch.nonzero(c).reshape(-1).tolist())
    for c in adjacency
  ]

def flatten(nested) :
  '''Works with 2d lists'''
  return list(itertools.chain(*nested))

def ltom(adjal, n=None) :
  if not n :
    n = 1 + max(flatten(adjal))

  adjacency = torch.zeros((n, n), dtype=torch.float)
  for i, edges in enumerate(adjal) :
    adjacency[i, edges] = 1.

  return adjacency

def permute(adjacency, features) :
  n = len(features)
  lg.debug('permute: len(features): %d', n)

  p = list(range(n))
  random.shuffle(p)
  lg.debug('permute: p: %s', p)

  lg.debug('permute: input adjacency: %s', adjacency)
  adjacency = mtol(adjacency)
  lg.debug('permute: mtol adjacency: %s', adjacency)
  adjacency = [[
    p.index(i) for i in adjacency[j]
  ] for j in p
  ]
  lg.debug('permute: dereference shuffle, adjacency: %s', adjacency)
  adjacency = ltom(adjacency, n)
  lg.debug('permute: ltom adjacency: %s', adjacency)
  lg.debug('permute: adjacency: size:%s, dtype:%s',
           adjacency.size(), adjacency.dtype)

  features = features[p]
  lg.debug('permute: features: size:%s, dtype:%s',
           features.size(), features.dtype)

  return adjacency, features

class gr_dataset(Dataset) :
  '''
For each __getitem__ 
return A, X, 
where A is Adjacency matrix (NvxNv)
      X is vertices (Nvx2); 
and   Nv is 4*num_rectangles

Presently num_rectangles=1 (by default)

TODO: 1. schema for padding
      2. parameter num_rectangles
  '''
  def __init__(self,
               ds_length=65536, # length of dataset
               permute=False,
               transform_adj=None,
               transform_v=None
  ) :
    super(gr_dataset, self).__init__()
    self.ds_length = ds_length
    self.permute = permute
    self.transform_adj = transform_adj
    self.transform_v = transform_v

  def __len__(self) :
    return self.ds_length

  def __getitem__(self, index) :
    adjacency, features = ADJACENCY, rand_verts()

    if self.permute :
      adjacency, features = permute(adjacency, features)

    if self.transform_adj :
      features = self.transform_adj(adjacency)

    if self.transform_v :
      features = self.transform_v(verts)

    # return load_adj(adjacency), features
    return adjacency, features

def is_int(s) :

  try:
    int(str(s))
  except Exception:
    return False

  return True

def draw(seq, scale=None, image=None) :
  '''Draw a line through the input SEQ of length 4*num_rectangles and
cycle defined as indices 0,1,3,2

  '''
  seq = (seq.numpy()
         if hasattr(seq, 'numpy')
         else np.array(seq)
  )
  
  assert seq.shape[0] % 4 == 0
  assert scale or image

  import cv2 as cv

  if not image :
    if is_int(scale) :
      scale = (scale, scale, 1)
      
    image = np.zeros(scale, dtype=np.uint8)

  if not scale :
    scale = image.shape

  seq = seq.reshape(-1, 4, 1, 2)
  seq[:, 2:] = np.flip(seq[:,2:], 1)
  seq *= np.array(scale[:2])
  seq = seq.astype(np.int32)
  cv.polylines(image, seq, True, (255,))

  return image

if __name__ == '__main__' :
  import time
  from datetime import timedelta as delT
  import cv2 as cv

  lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')

  N = (1 << 16) # ~65K
  dataloader = torch.utils.data.DataLoader(
    gr_dataset(ds_length=N, permute=True),
    batch_size=1024,
    shuffle=False # not required, __getitem__ is independent of index
  )

  lg.info('Testing for generating N:%dK images.', N // 1000)
  start = time.time()
  lg.info('Start' )
  

  # seq = None
  # for i, (A, X) in enumerate(dataloader) :
  #   if i == 0 :
  #     lg.info((A.size(), A.dtype, A[0]))
  #     lg.info((X.size(), X.dtype, X[0]))
  #     seq = X[:4]

  # end = time.time()
  # lg.info('End: %s', delT(seconds=(end-start)))
  # # takes 24s for N=1048K without permute on my dabba
  # # takes 17s for N=65K with permute on my dabba

  # cv.imwrite('/home/bvr/tmp/draw.png', draw(seq, 256))
  
def drawRec(A,X, name=""):

  lg.info((A.size(), A.dtype, A[0]))
  lg.info((X.size(), X.dtype, X[0]))
  seq = X[:4]

  cv.imwrite(name +'.png', draw(seq, 256))