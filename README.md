# DistributeLearningByGrpc

여러곳에 분산되어 저장되어 있는 data들을 사용하여 딥러닝 모델을 사용하고 평가 하는 것이 이 프로젝트의 목적이다.  
사용할 분산 저장되어 있는 data는 그대로 외부로 노출되면 민감한 정보이기 때문에 외부로 노출 되는 상황에서 이 data가 원래 어떤 data인지 어떤 의미를 가지는지 유추 할수 없어야 한다.  
위 조건을 만족시키기 위해, data가 저장되어 있는 remote server에서 딥러닝 model의 일부 layer를 통과 시켜 data가 어떤 의미를 가지는지 숨기고 추가적으로 중앙(main server)에서 실행될 학습량을 줄일 수 있다. 

Data가 저장되어 있고 딥러닝 model의 일부 layer를 통과 시켜 main server에 전송하는 server  ->  **remote server**  
Remote server들에서 받은 data를 merge해 나머지 model의 layer를 완료하여 최종 모델을 생성하고 평가하는 server  ->  **main server**

#### 1. Model Architecture

![12321](https://user-images.githubusercontent.com/68216852/119232348-e7ca1580-bb5f-11eb-91eb-2535691db9fc.png)


#### 2. Remote server(Grpc server)


#### 3. Main Server(Grpc call)


#### 4. Running Video
