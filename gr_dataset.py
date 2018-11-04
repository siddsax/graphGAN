## Graphical Rectangle Dataset with
import torch
from torch.utils.data import Dataset

import logging as lg
import numpy as np

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
               ds_length=65536 # length of dataset
  ) :
    super().__init__()
    self.ds_length = ds_length

  def __len__(self) :
    return self.ds_length

  def __getitem__(self, index) :
    return torch.FloatTensor(ADJACENCY), rand_verts()

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
    gr_dataset(ds_length=N),
    batch_size=1024,
    shuffle=False # not required, __getitem__ is independent of index
  )

  lg.info('Testing for generating N:%dK images.', N // 1000)
  start = time.time()
  lg.info('Start' )
  
  seq = None
  for i, (A, X) in enumerate(dataloader) :
    if i == 0 :
      lg.info((A.size(), A.dtype, A[0]))
      lg.info((X.size(), X.dtype, X[0]))
      seq = X[:4]

  end = time.time()
  lg.info('End: %s', delT(seconds=(end-start)))
  # takes 24s for N=1024k on my dabba

  cv.imwrite('/home/bvr/tmp/draw.png', draw(seq, 256))
  
