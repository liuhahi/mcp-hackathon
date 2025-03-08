import textarena as ta
 
model_name = "Standard GPT-4o LLM"
model_description = "Standard OpenAI GPT-4o model."
email = "liuzeyan1204@gmail.com"


# Initialize agent
agent = ta.agents.OpenRouterAgent(model_name="gpt-4o") 


env = ta.make_online(
    env_id="SpellingBee-v0", 
    model_name=model_name,
    model_description=model_description,
    email=email
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset()

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)


rewards = env.close()