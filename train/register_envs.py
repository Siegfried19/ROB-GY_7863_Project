from gymnasium.envs.registration import register

register(
    id="Go2WalkingGround-v0",
    entry_point="go2_env:Go2EnvMoonWalk",  
)



register(
    id="Go2FlyingingGround-v0",
    entry_point="go2_env:Go2EnvMoonFly",  
)