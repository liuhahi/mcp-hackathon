import textarena as ta
import os
import boto3, json, re
os.environ["AWS_ACCESS_KEY_ID"]="AKIAUC655UKYUSSDSM37"
os.environ["AWS_SECRET_ACCESS_KEY"]="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"
os.environ["AWS_DEFAULT_REGION"]="us-west-2"
region = "us-west-2"
system_prompt='You are a helpful assistant.'


model_name = "King Dedede abc"
model_description = "King Dedede abc"
email = "liuzeyan1204@gmail.com"

# STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format. Please make sure to use squared brackets to indicate the action you want to take."

STANDARD_GAME_PROMPT = """\
You are an AI agent playing a game called "Spelling Bee." Here are the rules:
1. You will receive a set of allowed letters. You must only use these letters to form your word.
2. Your word must be an English word recognized by the game's dictionary.
3. Each new word's length must be at least as long as the previous word's length (if any).
4. You cannot repeat a word that has already been used in this game.
5. You must always wrap your answer (the word) in square brackets. For example: [example].
6. Output only the word in square brackets, with no additional text, punctuation, or explanation.
7. You should NOT submit a word containing illegal words.
8. You should NOT only submit a English word!!!
9. You should NOT submit a non-English word!!!

Follow these rules precisely. If there is no previous word, choose any valid word (at least 3 letters, if applicable).
"""

validation_prompt = """
Please validate if the action
"""

validation_prompt2 = """
 is a correct valid. Please answer <yes> or <no> in your answer without any explanations.
"""

client = boto3.client('bedrock-runtime', region_name=region)
model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

agent = ta.agents.AWSBedrockAgent(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-west-2",
        system_prompt=STANDARD_GAME_PROMPT,
        verbose=False,
    )


env = ta.make_online(
    env_id=["SpellingBee-v0"],
    model_name=model_name,
    model_description=model_description,
    email=email
)

# env = ta.make_online(
#     env_id=["SimpleNegotiation-v0"],
#     model_name=model_name,
#     model_description=model_description,
#     email=email
# )

print("------------Initialized environment----------")
env = ta.wrappers.LLMObservationWrapper(env=env)
print("------------Prepared env----------")
env.reset(num_players=1)
print("------------Reseted env----------")
done = False
while not done:
    player_id, observation = env.get_observation()
    flag = True
    while flag:
        action = agent(observation)
        print('action',action)

        act_validation_prompt = validation_prompt + action + validation_prompt2

        content = [{
            "text": act_validation_prompt
        }]

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=[
                {
                    "text": system_prompt
                }
            ],
            inferenceConfig={
                "maxTokens": 8192,
                "temperature": 0,
            },
        )
        resp = response["output"]["message"]["content"][0]["text"]
        if "yes" in resp:
            done, info = env.step(action=action)
            flag = False
        # else:

        print('done',done)
        # print('info',info)


env.close()
print(info)