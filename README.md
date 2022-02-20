# DistributeLearningByGrpc

### Goal

여러곳에 분산되어 저장되어 있는 data들을 사용하여 딥러닝 모델을 사용하고 평가 하는 것이 이 프로젝트의 목적이다.  

사용할 분산 저장된 data는 그대로 외부로 노출되면 민감한 정보이기 때문에 외부로 노출 되는 상황에서 이 data가 원래 어떤 data인지 어떤 의미를 가지는지 유추 할 수 없어야 한다.  

위 조건을 만족시키기 위해, data가 저장되어 있는 remote server에서 딥러닝 model의 일부 layer를 통과 시켜 data가 어떤 의미를 가지는지 숨기고 추가적으로 중앙(main server)에서 실행될 학습량을 줄일 수 있다. 


Data가 저장되어 있고 딥러닝 model의 일부 layer를 통과 시켜 main server에 전송하는 server  
->  **remote server**  
Remote server들에서 받은 data를 merge해 나머지 model의 layer를 완료하여 최종 모델을 생성하고 평가하는 server  
->  **main server**


>>프로젝트에서 사용한 dataset들은 업로드 하지 않았음

### Version

server linux version  
->Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-70-generic x86_64)

Protobuf version  
->libprotoc 3.13.0  

Tensorflow & keras version  


![123321](https://user-images.githubusercontent.com/68216852/154835586-9b0d65d0-1dcd-4833-b02c-284d9ecc9d21.png)


## 1. Model Architecture

![12321](https://user-images.githubusercontent.com/68216852/119232348-e7ca1580-bb5f-11eb-91eb-2535691db9fc.png)


#### Architecture 동작 순서
1. Remote Server에서 grpc server를 작동한다. ( grpc server는 "grpc call"이 들어오면 자신이 가지고 있는 data를 model의 일부 layer를 통과 시키고 그 값을 return 해주는 function이 내장되어 있다.
2. Main Server에서 grpc call을 remote server에 요청한다.
3. Grpc call을 받은 remote server는 내장된 function을 통해 자신이 가지고 있는 data를 function에 넣고 반환 값을 다시 main server에 전송한다.
4. Main server에서 모든 remote server에 대한 return 값이 도착하면 모든 return 값을 merge 하여 딥러닝 model의 나머지 부분 학습을 진행한다.


## 2. Protobuf

```
syntax = "proto3";

service layer2_out{
        rpc request(DL_request) returns (DL_response){}
}

message DL_request{
        int32 state = 1;

}

message DL_response{
        bytes x_train = 1;
        bytes y_train = 2;
}
```

Grpc통신에서 사용할 message에 element과 service를 proto file에 정의한다. 이후 proto file 을 컴파일하여 grpc관련 function을 생성할 때 참조하는 service_pb2파일과 service_pb2_grpc 파일을 생성한다. 


## 3. Remote server(Grpc server)


data들이 저장되어 있는 server이며 grpc server을 실행시켜 main server로 부터 call을 기다리는 server.

### Return function
```
class layer2_out(service_pb2_grpc.layer2_outServicer):

    def request(self, request, context):
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=10)

        EPOCHS = 200
        rmse_result = list()

        data = pd.read_csv('1.csv')
        data =data.dropna(axis=1)

        data = data.drop(['DATE'], axis=1)
        data = data.rename(columns= {'L3008' : 'TC',
                                    'L3062' : 'HDL-C',
                                    'L3061' : 'TG',
                                    'L3068' : 'LDL-C'})

        x_train = data
        y_train = x_train.pop('LDL-C')

        model = makemodel.test_base(x_train)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

        mergedmodel = Model(inputs=model.input, outputs=model.get_layer('layer2').output)


        x_train_passed = mergedmodel.predict(x_train)


        bx = np.array(x_train_passed,dtype=np.float32).tobytes()
        by = np.array(y_train,dtype=np.float32).tobytes()
        return service_pb2.DL_response(x_train=bx,y_train=by)
```
proto파일을 컴파일하여 생성된 pb2_grpc에 선언되어있는 outServicer를 상속받아 request function을 생성한다. 이 function안에서 remote server에 저장되어있는 data를 processing 하여 model의 layer2 까지 통과하고 통과한 x_train 값과 y_train 값을 return 한다. main server에 return 값을 전송하기 위해 data 형태를 byte 형식으로 바꾼다. 


### Run Grpc Server  
```
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[('grpc.max_send_message_length', 1024 * 1024 * 200),('grpc.max_receive_message_length', 1024 * 1024 * 200),],)
    service_pb2_grpc.add_layer2_outServicer_to_server(layer2_out(), server)
    server.add_insecure_port('172.25.244.2:50051')
    server.start()
    server.wait_for_termination()
```  
Grpc server를 구성하고 실행하는 function. server를 구성할때 옵션 값 options=[('grpc.max_send_message_length', 1024 * 1024 * 200),('grpc.max_receive_message_length', 1024 * 1024 * 200),]을 통해 Grpc 통신을 통해 전송할 수 있는 message의 byte 수를 조정한다.

## 4. Main Server(Grpc call)


remote server들에게 data를 요청하는 grpc call을 보내고 받은 return 값들을 merge하며 나머지 model를 학습하는 server


### Call function
```  
def run1():
        with grpc.insecure_channel('ip:port', options=[('grpc.max_send_message_length', 1024 * 1024 * 300), ('grpc.max_receive_message_length', 1024 * 1024 * 300), ],) as channel:
                stub = service_pb2_grpc.layer2_outStub(channel)
                response = stub.request(service_pb2.DL_request(state=1))
                xb_train = np.frombuffer(response.x_train, dtype=np.float32)
                yb_train = np.frombuffer(response.y_train, dtype=np.float32)

                x_train = xb_train.reshape(-1, 128)
                y_train = pd.Series(yb_train, name='LDL-C')
                return x_train, y_train
```

Remote server에게 Grpc call을 보내고 call에 대한 return 값을 return 하는 function이다. remote server에서 전송시 byte 형태로 바꾸어서 전송하여서 다시 data type을 바꾸고 data type를 바꾸는 과정에서 np.array의 shape이 1차원으로 바꾸기 때문에 이에 대해서 shape을 조정하는 과정을 거친다.

### Bring Testset function
```
def testset():
        optimizer = optimizers.Adam(0.01)
        early_stop = callbacks.EarlyStopping(
    monitor='val_mean_squared_logarithmic_error', patience=10)

        EPOCHS = 200
        rmse_result = list()

        data = pd.read_csv('testset.csv')
        data = data.dropna(axis=1)

        data = data.drop(['DATE'], axis=1)
        data = data.rename(columns= {'L3008': 'TC', 'L3062': 'HDL-C', 'L3061' : 'TG', 'L3068' : 'LDL-C'})

        x_test = data
        y_test = x_test.pop('LDL-C')

        model = makemodel.test_base(x_test)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_logarithmic_error'])

        mergedmodel = Model(inputs=model.input, outputs=model.get_layer('layer2').output)
        mergedmodel.summary()

        x_test_passed = mergedmodel.predict(x_test)
        return x_test_passed, y_test
```

model을 test하는데에 필요한 test_data를 만드는 function이다. 

## 5. Running Video

https://user-images.githubusercontent.com/68216852/119236618-6aa89b80-bb73-11eb-89bd-8c46c4e391a6.mp4


## 6. Test Case

논문에서 사용한 Test case별로 구성한 코드는 test_folder에 기록되어 있으며 사용한 모든 데이터는 기록에는 모두 제외 했음(local에는 데이터가 포함되어 저장되어 있음: xxx.xxx.xxx.2 PC에 /SplitLearning/include_data).


#### 각 폴더의 의미
local  : 일반적인 머신러닝 환경으로 모든 data를 하나의 host에서 수행  
case_1 : remote host에서 x_train 값에 일정한 vector를 곱하고 main server에 전송  
case_2 : remote host에서 일부 layer을 predict 함수로 학습하지않고 통과(필터)한 값을 main server에 전송   
case_3 : remote host에서 일부 layer을 학습시키고 통과 한 값을 main server에 전송  
cnn    : 위의 과정을 cifar10 data을 가지고 cnn 학습 수행  


#### 논문  
D. Lee, J. Lee, H. Jun, H. Kim and S. Yoo, "Triad of Split Learning: Privacy, Accuracy, and Performance," 2021 International Conference on Information and Communication Technology Convergence (ICTC), 2021, pp. 1185-1189, doi: 10.1109/ICTC52510.2021.9620846.
