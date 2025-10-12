from gym.envs.registration import register

register(
    id="Go2WalkingGround-v0",
    entry_point="go2_env:Go2Env",  
)



register(
    id="Go2JumpingGround-v0",
    entry_point="go2_env:Go2EnvMoon",  
)