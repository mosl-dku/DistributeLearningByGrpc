# DistributeLearningByGrpc

여러곳에 분산되어 저장되어 있는 data들을 사용하여 딥러닝 모델을 사용하고 평가 하는 것이 이 프로젝트의 목적이다.  

사용할 분산 저장된 data는 그대로 외부로 노출되면 민감한 정보이기 때문에 외부로 노출 되는 상황에서 이 data가 원래 어떤 data인지 어떤 의미를 가지는지 유추 할 수 없어야 한다.  

위 조건을 만족시키기 위해, data가 저장되어 있는 remote server에서 딥러닝 model의 일부 layer를 통과 시켜 data가 어떤 의미를 가지는지 숨기고 추가적으로 중앙(main server)에서 실행될 학습량을 줄일 수 있다. 


Data가 저장되어 있고 딥러닝 model의 일부 layer를 통과 시켜 main server에 전송하는 server  
->  **remote server**  
Remote server들에서 받은 data를 merge해 나머지 model의 layer를 완료하여 최종 모델을 생성하고 평가하는 server  
->  **main server**

## 1. Model Architecture

![12321](https://user-images.githubusercontent.com/68216852/119232348-e7ca1580-bb5f-11eb-91eb-2535691db9fc.png)

#### Architecture 동작 순서
1. Remote Server에서 grpc server를 작동한다. ( grpc server는 "grpc call"이 들어오면 자신이 가지고 있는 data를 model의 일부 layer를 통과 시키고 그 값을 return 해주는 function이 내장되어 있다.
2. Main Server에서 grpc call을 remote server에 요청한다.
3. Grpc call을 받은 remote server는 내장된 function을 통해 자신이 가지고 있는 data를 function에 넣고 반환 값을 다시 main server에 전송한다.
4. Main server에서 모든 remote server에 대한 return 값이 도착하면 모든 return 값을 merge 하여 딥러닝 model의 나머지 부분 학습을 진행한다.


## 2. Remote server(Grpc server)

data들이 저장되어 있는 server이며 grpc server을 실행시켜 main server로 부터 call을 기다리는 server.


## 3. Main Server(Grpc call)

remote server들에게 data를 요청하는 grpc call을 보내고 받은 return 값들을 merge하며 나머지 model를 학습하는 server


## 4. Running Video

https://user-images.githubusercontent.com/68216852/119236618-6aa89b80-bb73-11eb-89bd-8c46c4e391a6.mp4
