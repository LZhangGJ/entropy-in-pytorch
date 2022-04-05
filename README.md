# entropy-in-pytorch
A entropy hub by pytorch</br>
<b> l: signal </br>
  m: windows length </br>
  r: threshold </br>
  ```
  e.g: 
  import torch
  l = torch.randn(1000) /n
  m = 2 
  r = torch.std(l)*0.25
  sampEn(l,m,r)
  ```
