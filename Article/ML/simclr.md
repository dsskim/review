# SimCLR in PyTorch

- SimCLR(Simple Framework for Contrastive Learning of Visual Representations)
- Self-supervised learning의 SOTA 방법

## SimCLR contribution
- unsupervised learning는 강한 augmentation을 통해 이득을 얻음
- base encoder 이후에 학습 가능한 MLP를 도입하면 학습된 표현의 품질 향상
- contrastive cross-entropy loss 학습은 normalized embedding과 적절하게 조절된 파라미터에 이점을 얻음(?)
- Contrastive learning는 supervised learning에 비하여 더 큰 배치 사이즈와 더 긴 학습에 이점을 가짐. supervised learning와 마찬가지로, network이 deeper 및 wider할 수록 좋음

## SimCLR의 4가지 구성요소

![framework](images/simclr_01.png)

### 1. stochastic data augmentation module 
- 주어진 데이터를 무작위로 변환하여 두개의 상관된 데이터로 변환하는 과정
- 같은 데이터에서 뽑았으면 positive pair, 다른 데이터에서 뽑았으면 negative pair로 간주
- 저자는 3개의 간단한 augmentation 적용: Random Cropping, Random Color Distortions, Random Gaussian Blur

### 2. base encoder f
- augmented 데이터로부터 특징 벡터를 추출하는 Network

### 3. small neural network projection head g
- Contrastive loss를 적용하기 위한 공간에 base encoder로 부터 나온 특징 벡터를 맵핑

### 4. Contrastive Loss function

![loss](images/simclr_02.png)
