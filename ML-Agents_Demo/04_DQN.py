import numpy as np
import random
import copy
import datetime
import platform
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                                                    import EngineConfigurationChannel
import mlagents_envs.rpc_utils as rpc_utils
from mlagents_envs.exception import UnityObservationException
# DQN은 한 번 모델을 학습할 떄 리플레이어 메모리에서 일정 개수만큼의 경험을 랜덤하게 추출해 미니 배치 학습을 수행.

_original_observation_to_np_array = rpc_utils._observation_to_np_array


def _observation_to_np_array_safe(obs, expected_shape=None):
    try:
        return _original_observation_to_np_array(obs, expected_shape)
    except UnityObservationException:
        if obs.compression_type != rpc_utils.COMPRESSION_TYPE_NONE:
            img = rpc_utils.process_pixels(
                obs.compressed_data,
                obs.shape[2],
                list(obs.compressed_channel_mapping),
            )
            if list(img.shape) == [obs.shape[1], obs.shape[2], obs.shape[0]]:
                return img
        raise


rpc_utils._observation_to_np_array = _observation_to_np_array_safe
                                           
state_size = [3*2, 64,84] # 그리드 상황을 알 수 있는 시각적 관측 정보(높이 64 ,너비 84, 채널 3인 RGB 이미지. -> 목적지 관측 정보와 시각적 관측 정보를 합쳐 하나의 상태로 만들기 위해 rgb이미지를 2번 중첩하여 채널을 6으로 만든 후 목적지 관측 정보에 따라 각 채널을 전처리)
action_size = 4 # DQN 네트워크의 출력으로 사용할 행동의 크기. [정지, 위, 아래, 왼쪽, 오른쪽]이 원래 크기이나, 학습의 효율성을 위해 정지 행동을 고려하지 않음

load_model = False # 미리 학습된 모델을 불러오지 않음
train_mode = True # 학습 모드 -> FALSE로 설정하면 학습을 진행하지 않고 네트워크 연산 결과에 따라 행동을 선택
batch_size = 32 # DQN 네트워크를 학습할 때 사용되는 미니 배치의 크기. 리플레이 메모리에서 무작위로 추출된 경험 샘플의 수를 나타냄. 일반적으로 32 또는 64로 설정하는 경우가 많음. 너무 작은 배치는 학습이 불안정해질 수 있고, 너무 큰 배치는 학습 속도를 저하시킬 수 있음.
mem_maxlen = 10000 # 리플레이 메모리의 최대 크기
discount_factor = 0.9 # 학습 수횅 시 얼마나 미래의 보상을 고려할 지 결정하는 감가율(0~1사이) -> 이 값이 클수록 미래의 보상을 많이 고려하여 학습을 수행
learning_rate = 0.00025 # 네트워크 학습 수행 속도 -> 너무 작으면 학습 속도가 느려지고, 너무 크면 학습이 불안정하게 수행됨

run_step = 50000 if train_mode else 0 # 총 몇번의 스텝 동안 학습을 수행할지 -> 평가 모드일 떄는 0
test_step =5000 # 학습이 끝나고, 혹은 평가 모드에서 몇 스텝 동안 테스트할 지 결정
train_start_step =5000 # 학습 시작 전 리플레이 메모리에 충분한 데이터를 모으기 위해 몇 스텝동안 임의의 행동으로 게임을 진행할 것인지 결정
target_update_step = 500 # 타깃 네트워크를 몇 스텝 주기로 업데이트할 지 결정

print_interval = 10
save_interval = 100

# DQN의 기법 중 하나인 엡실론 그리디 기법 관련 파라미터
epsilon_eval = 0.05# 평가 모드에서의 엡실론 값. 0.05->평가 모드에서 5%의 확률로 무작위 행동을 수행 -> 0으로 설정하면 특정 경우에 목적지에 도달하지 않고 반복 행동을 하며 특정 위치를 벗어나지 못하는 문제가 존재하기에, 이를 보완하기 위함
epsilon_init = 1.0 if train_mode else epsilon_eval # 엡실론 초기값 -> 에피소드 초반에 에이전트가 탐험을 하도록 1로 설정.
epsilon_min = 0.1 # 엡실론 최소값 -> 에이전트가 특정 상태에서 매번 동일한 행동을 하는 것을 막기 위함
explore_step = run_step * 0.8 # 엡실론이 감소하는 구간. 학습 스텝의 80%동안 엡실론이 감소하여 에이전트가 탐험을 하도록 설정. 이후에는 엡실론이 최소값으로 유지되어 어느 정도의 탐험이 계속 이루어지도록 함
epsilon_delta = (epsilon_init - epsilon_min)/ explore_step if train_mode else 0. # 한 스텝당 감소하는 엡실론의 변화량 -> 첫 스텝에서 1로 시작하고, START_TRAIN_STEP 지점부터 학습이 진행되는 EXPLORE_STEP 구간 동안 엡실론 값이 서서히 감소하며, epsilon_min 에 도달하면 학습이 끝날 떄 까지 그 값을 유지하는 전략을 사용

VISUAL_OBS = 0 # 그리드월드 환경에서 시각적 관측의 인덱스 상수. 시각적 관측은 RGB 이미지 형태로 제공되며, 에이전트가 주변 환경을 인식하는 데 사용됨
GOAL_OBS = 1 # 목적지 관측의 인덱스 상수
VECTOR_OBS = 2 # 벡터 관측의 인덱스 상수.
OBS = VISUAL_OBS # 그리드월드 환경에서 어떤 관측을 사용할 지 정하는 변수. 시각적 관측을 사용할 것이므로 VISUAL_OBS로 설정

game = "ML_Agents.exe"
os_name = platform.system() # 현재 운영체제 정보 가져오기
if os_name == 'Windows' :
    env_name = str((Path(__file__).resolve().parent / "ENV/GridWorld_Windows" / game).resolve())
elif os_name == 'Darwin' :
    env_name = f"../ENV/{game}_{os_name}"

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"../saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/202605102011"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          

# 모델 클래스
class DQN(torch.nn.Module):
    def __init__(self, **kwargs): # 네트워크 초기화 함수. 네트워크를 구성하는 레이어를 정의. model 객체를 데이터와 함께 호출할 때 실행
        super(DQN, self).__init__(**kwargs)
        
        # 컨볼루션 레이어 생성(dqn 논문에서 제안된 아키텍처를 기본으로 구성) -> Conv2d 파라미터는 (입력 채널 수, 출력 채널 수, 커널 크기, 스트라이드)
        # 논문에서 정의된 구조는 32(8*8)-64(4*4)-64(3*3)->flat->512units->action size 큐
        self.conv1 = torch.nn.Conv2d(in_channels= state_size[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((state_size[1] - 8)//4+1, (state_size[2] - 8)//4+1)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2+1, (dim1[1] - 4)//2+1)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1+1, (dim2[1] - 3)//1+1)
                
        # 컨볼루션을 통해 나온 결과를 fully connected layer의 입력으로 사용하기 위해 ,결과를 1차원으로 바꾸어야 하므로 flatten 레이어 정의
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], 512) # torch.nn.Linear를 사용해 fully connected layer를 만들고, 마지막 출력인 큐를 정의. Linear 파라미터는 (입력 차원, 출력 차원)
        self.q = torch.nn.Linear(512, action_size)
        
    def forward(self, x): # 기존에 선언한 레이어를 통해 네트워크 입력 값에 대한 큐 함수를 계산하는 함수. 입력 x를 init함수에서 정의한 레이어들에 순서대로 통과시켜 큐 함수를 계산.
        x = x.permute(0,3,1,2)  # permute를 사용해 입력 이미지의 차원을 변환 (유니티와 파이토치의 이미지 차원이 다르기 떄문. 유니티에서는 이미지 Shape가 (Height, Width, Channel), 파이토치는 (Channel, Height, Width)순서.)
        # 각 단계별 네트워크 연산 -> permute를 통과한 데이터를 각 컨볼루션 레이어에 통과시킨다.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #flat 함수를 사용하여 결과를 1차원으로 만들고, Linear 레이어인 fc1을 통과시켜 큐 함수값을 도출. 이떄, 각 히든 레이어에서는 ReLU활성화 함수를 사용
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        return self.q(x) # 입력이 -1 ~ 1까지의 값을 갖도록 정규화
    
  # 에이전트 클래스  
class DQNAgent:
    def __init__(self): # 모델 클래스에서 정의한 DQN 모델을 가져와서 행동을 결정하거나 큐 함수 값을 예측할 떄 사용하는 network를 생성.
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network) # 학습에 필요한 타겟값을 계산하는 타겟 네트워크 생성. copy 라이브러리에서 deepcopy 함수를 사용하여 network와 동일한 구조 및 가중치를 갖는 네트워크로 생성.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate) # 네트워크의 가중치를 업데이트하는 데 사용되는 옵티마이저 생성. Adam 옵티마이저를 사용하여 네트워크의 매개변수를 업데이트하며, 학습률은 learning_rate로 설정
        self.memory = deque(maxlen = mem_maxlen) # 리플레이 메모리 역할의 변수. 데크로 데이터를 저장하면 mem_maxlen보다 더 많은 데이터를 저장할 떄 자동으로 가장 오래된 데이터를 삭제한 후 새로운 데이터를 추가할 수 있음.
        self.epsilon = epsilon_init 
        self.writer = SummaryWriter(save_path)
        
        if load_model == True: # 저장된 모델을 사용할 경우.
            print(f"...Load Model From {load_path}/ckpt...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=  device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    def get_action(self, state, training = True): # 엡실론 그리디 기법에 따라 행동을 결정하도록 하는 함수
        # 네트워크 모드 결정
        self.network.train(training) # 네트워크 학습 모드 또는 평가 모드 설정. 만약 batch_normaliozation 레이어나 dropout레이어가 있을 경우, 학습 모드일 떄와 평가 모드일 때 다르게 작동함.
        epsilon = self.epsilon if training else epsilon_eval # 학습 모드일 떄는 엡실론 값을 그대로 사용.
        
        # 엡실론 그리디 기법에 따라 행동을 결정하는 코드. 엡실론 값이 0~1 사이의 랜덤 값보다 크면 에이전트는 엡실론 그리디에 따라 랜덤 행동을 시작.
        if epsilon > random.random():  
            action = np.random.randint(0, action_size, size = (state.shape[0], 1)) # 그리디월드의 행동 크기는 4이므로, randint 함수를 사용해 0~3 사이 중 하나의 값을 선택. 
        else : # 랜덤 값 > 엡실론 값이면 네트워크 결과에 따라 행동 선택
            q = self.network(torch.FloatTensor(state).to(device)) # 네트워크에 파라미터를 넣어줄 때는 상태를 FloatTensor 형태로 변환하여 넣어주어야 함.
            action = torch.argmax(q, axis = -1, keepdim = True).data.cpu().numpy() # 네트워크 결과(q)에서 가장 큰 큐 함수값을 갖는 인덱스를 action으로 선택함.
        return action
    
    def append_sample(self, state, action, reward, next_state, done) :  # 리플레이 메모리에 데이터를 추가하는 함수.(상태, 행동, 보상, 다음 상태, 게임 종료 여부)
        self.memory.append((state, action, reward, next_state, done))
        
    def train_model(self): # 학습 수행 함수
        batch = random.sample(self.memory, batch_size) #batch_size만큼 무작위로 데이터를 추출해 batch에 저장.
        state = np.stack([b[0] for b in batch], axis = 0) # 각각의 상태, 행동, 보상, 다음 상태, 게임 종료 여부를 넘파이 배열 벡터로 변환하고, 0번쨰 축을 기준으로 배치 차원만큼 쌓는다.
        action = np.stack([b[1] for b in batch], axis = 0)
        reward = np.stack([b[2] for b in batch], axis = 0)
        next_state = np.stack([b[3] for b in batch], axis = 0)
        done = np.stack([b[4] for b in batch], axis = 0)
        
        # map 함수를 사용하여 각 파라미터 데이터드를 각각 FloatTensor 형태로 변환 후 디바이스 메모리에 올린다.
        state, action, reward, next_state, done = map(lambda x : torch.FloatTensor(x).to(device), [state, action, reward, next_state, done])\
        
        # 현재 행동에 대한 q값만 취득해야 함
        eye = torch.eye(action_size).to(device) #action_size 크기만큼 action에 대해 원핫 인코딩 
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims = True) # state에 대한 일반 네트워크의 출력에 one_hot_action을 곱하면 현재 행동의 인덱스에 해당하는 큐 함수 값만 남고 나머지는 0이 됨 - >sum으로 현재 행동에 대한 큐 함수값을 구해 q에 저장.
        
        with torch.no_grad(): # 네트워크 학습을 위해 필요한 타겟 값을 구하는 과정이 네트워크 업데이트에 관여되지 않도록 with torch.no_grad()구문 안에서 그라디언트 추적이 되지 않도록 함
            next_q = self.target_network(next_state) # 다음 상태에 대한 큐 값
            target_q = reward + next_q.max(1, keepdims = True).values * ((1-done) * discount_factor) # next_q 값에 대해 각 행동에 대한 큐 함수 값 중 가장 큰 값
            
        loss = F.smooth_l1_loss(q, target_q) # q, 후버 로스 계산
        
        self.optimizer.zero_grad() # 옵티마이저의 그라디언트를 0으로 초기화
        loss.backward()  # 역전파를 통해 그라디언트 값을 계산
        self.optimizer.step() # 모델 파라미터 값 업데이트
        
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta) # 엡실론 그리디에 따른 학습 진행을 위해 엡실론 값 서서히 감소
        
        return loss.item()
        
        
    def update_target(self): # 타겟 네트워크 업데이트 함수
        self.target_network.load_state_dict(self.network.state_dict()) # state_dict로 일반 네트워크를 불러온 후 load_stat_dict를 통해 타겟 네트워크에 파라미터를 복제
    
    def save_model(self): # 네트워크 모델 저장 함수
        print(f"...Save Model to {save_path}/ckpt...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }, save_path+'/ckpt')
        
    def write_summary(self, score, loss, epsilon, step): # 학습 과정에서의 지표를 기록하는 함수
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)

#메인 함수
        """
            DQN 알고리즘 동작 과정
            유니티 환경 설정 & 브레인 정의 -> 행동 선택 -> 유니티에서 행동 수행 -> 유니티 환경으로부터 다음 상태, 보상, 게임 종료 등 정보 취득 -> 학습 -> 특정 스텝마다 타겟 네트워크 업데이트 -> 진행상황 출력 및 특정 스텝마다 모델 저장 -> 행동 선택 ->...
            
            1. 유니티 환경에 대한 설정 후 브레인 정의
            2. 에이전트가 취할 행동을 선택하고, 유니티 환경의 브레인을 통해 해당 행동을 전송해 유니티 환경 내부의 에이전트가 행동을 수행하도록 함
            3. 에이전트가 행동을 취하면 Python 코드는 유니티를 통해 다음 상태, 보상 ,게임 종료 여부 등을 전달받음
            4. 위 정보들을 이용해 네트워크 학습 수행
            5. 특정 스텝마다 상태 업데이트
            6. 반복
        
        """
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel() # 유니티 환경의 설정을 제어하기 위한 사이드 채널 생성 -> 사이드 채널은 유니티 환경의 time-scale이나 해상도, 그래픽 퀄리티 등을 수정할 떄 사용
    
    env = UnityEnvironment(file_name = env_name, side_channels=  [engine_configuration_channel])
    env.reset() # 유니티 환경 초기화
    
    #유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0] # keys()는 모든 behavior 정보를 갖고 있으며, list로 변환하여 첫 번째 요소를 behavior_name에 저장
    
    spec = env.behavior_specs[behavior_name] # behavior_name 키를 통해 spec 정보를 얻음(observation과 action에 대한 정보가 담긴 객체)
    
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0) # 유니티 환경의 time-scale을 12로 설정 -> time-scale이 12이면, 유니티 환경이 실제 시간보다 12배 빠르게 진행됨. 학습 속도를 높이기 위해 time-scale을 높이는 경우가 많음.
    
    dec, term = env.get_steps(behavior_name) # 유니티 환경에서 행동을 요청한 상태인지, 마지막 상태인지 확인하는 함수. decision_steps에는 행동을 요청한 상태의 정보(에피소드가 종료되지 않고 계속 진행 중인 상황)가 담긴 객체가 저장되고, terminal_steps에는 마지막 상태의 정보가 담긴 객체가 저장됨.
    
    agent = DQNAgent() # 에이전트 객체 생성
    
    losses, scores, episode, score = [],[],0,0
    for step in range(run_step + test_step) : # 학습 모드 스텝 수 + 테스트 모드 스텝 수만큼 반복
        if step == run_step:
            if train_mode : 
                agent.save_model() # 학습이 끝난 시점에 모델 저장
            print("TEST START")
            train_mode = False # 테스트 모드로 전환
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0) # 테스트 모드에서는 time-scale을 1로 설정하여 실제 시간과 동일하게 진행되도록 함.
    
    """
        dec.obs는 지정한 behavior 이름을 가진 모든 에이전트에 대한 모든 관측을 포함하는 튜플
        OBS는 시각적 인덱스인 0으로 설정했끼 떄문에 dec.obs[OBS]에는 시각적 관측 정보가 있ㄲ고, dec.obs[GOAL_OBS]에는 목적지 관측 정보가 있음.
        목적지가 + 일 때는 [[1,0]]이고, - 일 때는 [[0,1]]을 정보로 가진다.
        
        preprocess를 통해 시각적 관측 정보에 goal[0]][0]가 [0][1]을 각각 곱하여 concatenate 연산 -> 목적지가 plus일 때는 6채널 중 앞 4개 채널만 RGB 값으로 들어가고, 나머지 3개 채널에는 모두 0이 들어감
        반대로 목적지가 ex일 때는 뒤에 있는 3개 채널 값이 RGB로 들어가고, 앞의 3개 채널은 0이 들어감.
        """
    preprocess = lambda obs, goal : np.concatenate((obs * goal[0][0], obs*goal[0][1]), axis = -1) # 시각적 관측정보와 목적지 관측 정보를 전처리하는 람다함수.
    state = preprocess(dec.obs[OBS], dec.obs[GOAL_OBS]) # 시각적 관측과 목적지 관측을 전처리하여 하나의 상태로 만듦
    
    action = agent.get_action(state, train_mode) # 엡실론 그리디에 따라 행동 선택
    real_action = action + 1 #  행동이 0 ~ 3으로 선택되기 때문에 1을 더하여 0번 행동인 정지를 제외
    
    action_tuple = ActionTuple() # 행동을 유니티 환경에 전달하기 위한 ActionTuple 객체 생성
    action_tuple.add_discrete(real_action) # 행동을 ActionTuple 객체에 추가. add_discrete 함수를 사용하여 행동을 이산 행동으로 추가
    env.set_actions(behavior_name, action_tuple) # 행동을 유니티 환경에 설정. behavior_name에 해당하는 에이전트들에게 action_tuple로 정의된 행동을 설정
    env.step() # 유니티 환경에서 한 스텝 진행 
    
    dec, term = env.get_steps(behavior_name) # 스텝이 진행된 후 다시 행동을 요청한 상태인지, 마지막 상태인지 확인
    done = len(term.agent_id) > 0
    reward = term.reward if done else dec.reward # 에피소드가 종료된 상태에서는 term.reward를 사용하고, 그렇지 않은 상태에서는 dec.reward를 사용하여 보상을 얻음
    
    next_state = preprocess(term.obs[OBS], term.obs[GOAL_OBS]) if done else preprocess(dec.obs[OBS], dec.obs[GOAL_OBS]) # 다음 상태도 시각적 관측과 목적지 관측을 전처리하여 하나의 상태로 만듦
    score == reward[0]  #  그리드 월드 환경은 하나의 에이전트만 존재하기 때문에, term의 agent_id의 갯수를 보고 종료 여부를 파악 가능 -> 0이면 조욜가 아닌 상태, 1이면 종료인 상태
    
    if train_mode : 
        agent.append_sample(state[0], action[0], reward, next_state[0], [done]) # 리플레이 메모리에 경험 추가
    
    if train_mode and step > max(batch_size, train_start_step) : # dqn은 배치 학습을 위해 적어도 배치 사이즈 이상의 데이터가 핋요하기에, 충분한 데이터를 모아두고 학습을 해야 함
        loss = agent.train_model() # 함수로부터 반환된 손실함수 값 저장
        losses.append(loss) #  손실함수 값을 리스트에 저장

        if step % target_update_step == 0 :# 스텝이 target_update_step만큼 진행될 떄 마다 타겟 네트워크 업데이트
            agent.update_target()
    if done : # 에피소드가 종료된 경우
        episode+=1
        scores.append(score) # 에피소드마다 누적된 보상 저장
        score = 0 # 누적된 보상 초기화
        
        if episode % print_interval == 0 : 
            mean_score = np.mean(scores)
            mean_loss = np.mean(losses)
            agent.write_summary(mean_score, mean_loss, agent.epsilon, step) # 텐서보드에 지표 기록
            losses, scores = [], [] # 손실함수 값과 보상 값 초기화
            
            print(f"Episode : {episode} Step : {step} Score : {mean_score:.2f} Loss : {mean_loss:.4f} Epsilon : {agent.epsilon:.4f}") # 진행 상황 출력
        if episode % save_interval == 0 : # 에피소드가 save_interval만큼 진행될 때 마다 모델 저장
            agent.save_model()
    env.close()
        
    