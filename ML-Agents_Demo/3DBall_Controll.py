from mlagents_envs.environment import UnityEnvironment # 유니티 환경 라이브러리를 임포트. 이 라이브러리는 유니티와 상호작용하는 데 사용

if __name__ == '__main__':
    env = UnityEnvironment(file_name = './ENV') # 빌드된 환경 경로
    env.reset()  # 환경 초기화
    
    behavior_name = list(env.behavior_specs.keys())[0] # behavior 관련 정보를 behavior_name과 spec에 저장. 
    """
    behavior는 같은 브레인을 공유하는 에이전트들의 그룹이며, c#의 behavior parameters를 동일하게 사용하는 에이전트들.
    behavior_specs는 에이전트들의 행동이나 관측 정보들이 저장되어 있음.
    """
    print(f'name of behavior : {behavior_name}')
    spec = env.behavior_specs[behavior_name] # 해당 behavior의 이름을 저장하고, 그 이름에 대한 행동 및 관측 정보를 spec에 저장
    
    for ep in range(10) :# 에피소드 10번 반복
        env.reset()
        
        decision_steps, terminal_steps = env.get_steps(behavior_name) # 에이전트가 행동을 요청한 상태인지, 마지막 상태인지 확인
        """
        get_steps는 각 스텝에서 에이전트의 상태, 행동, 보상 등의 정보를 반환.
        해당 정보들이 다음 행동을 요청한 스텝일 때(아직 환경의 에피소드가 끝나지 않은 상태로 스텝이 진행 중일 때)-> 정보는 decision_step에 저장
        만약 에피소드가 끝난 마지막 스텝일 때 -> terminal_step에 정보들이 저장되고, decision_step에는 다음 에피소드의 첫 스텝 값이 저장됨
        """
        # 에이전트 하나를 기준으로 로그를 출력 
        tracked_agent = -1 # 추적할 에이전트의 아이디 (3DBall의 경우 총 12개의 에이전트가 존재하며, 현재는 한 에이전트의 정보만 출력할 거니까 에이전트 아이디를 따로 저장.
        done = False # # 한 에피소드가 마무리 되었는지 판단하는 변수
        ep_rewards = 0 # 해당 에피소드 동안 보상의 합을 저장할 변수
        
        while not done : # 에피소드가 끝날 때까지 반복
            if tracked_agent == -1 and len(decision_steps) >=1: # tracked_agent 정의
                tracked_agent = decision_steps.agent_id[0] # 첫 번째 아이디 사용
            
            action = spec.action_spec.random_action(len(decision_steps)) # 랜덤 액션 결정 : desicion_steps의 길이 ,즉 에이전트 총 숫자인 12개의 에이전트에 대한 행동을 모두 도출하고, 이를 action에 저장
            # 이때 action은 ActionTuple이라는 형태로 저장. ActionTuple은 유니티 ML-Agents에서 에이전트의 행동을 표현하는 데이터 구조
            # 12개의 에이전트가 모두 동일한 behavior parameter를 가지기 때문에 하나의 behavior_name에 대해서 모든 행동을 결정해주면 됨.
            # 만약 다른 behavior parameter를 가진 에이전트들이 존재한다면, 각각의 behavior_name에 대해서 별도로 행동을 결정해줘야 함.
            
            env.set_actions(behavior_name, action) # 행동 설정 : behavior_name에 해당하는 에이전트들에게 action을 설정
            env.step() # 환경 한 스텝 진행
            
            decision_steps, terminal_steps = env.get_steps(behavior_name) # 스텝 종료 후 다시 에이전트들의 상태를 가져옴
            
            if tracked_agent in decision_steps :
                ep_rewards+=decision_steps[tracked_agent].reward # 해당 에이전트가 decision_steps에 있다면, 즉 아직 에피소드가 끝나지 않았다면 보상을 누적
            
            if tracked_agent in terminal_steps :
                ep_rewards+=terminal_steps[tracked_agent].reward # 해당 에이전트가 terminal_steps에 있다면, 즉 에피소드가 끝났다면 보상을 누적
                done = True # 에피소드 종료 표시
            
        print(f'total reward for ep {ep} is {ep_rewards}') # 에피소드 동안 누적된 보상 출력
        
    env.close() # 환경 종료