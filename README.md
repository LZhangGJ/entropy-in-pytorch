# entropy-in-pytorch
A entropy hub by pytorch
<b> l: signal \n
  m: windows length \n
  r: threshold \n
 e.g: 
  l = torch.randn(1000)
  m = 2
  r = torch.std(l)*0.25
  sampEn(l,m,r)
