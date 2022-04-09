# entropy-in-pytorch
A entropy hub by pytorch</br>
now in working
Just in beginning</br>
A sample for sample entropy:</br>
<b> l: signal </br>
  m: windows length </br>
  r: threshold </br>
  ```
  e.g: 
  import torch
  l = torch.randn(1000) 
  m = 2 
  r = torch.std(l)*0.25
  sampEn(l,m,r)
  
  
  
  '''
  For batch Data (B,C,H)
  The return shape should be (B,C)
  '''
  l = torch.randn([5,5,20])
  sampEn_batch(l,m,r)
  
  ```
