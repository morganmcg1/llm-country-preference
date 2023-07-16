import os
import openai
import json
import wandb
import pandas as pd
from tqdm import tqdm
import logging
from fastcore.parallel import parallel
from tenacity import retry, stop_after_attempt, wait_fixed, after_log

logger = logging.getLogger(__name__)

functions = [{
        "name": "print_sentiment",
        "description": "A function that prints the given sentiment of a piece of text",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "The sentiment of the text.",
                },
            },
            "required": ["sentiment"],
        }
    }]

api_key = os.environ["OPENAI_API_KEY"]

# Define your rate limit parameters
max_attempts = 5  # change as needed
wait_time = 45  # change as needed
SYSTEM_PROMPT = "You are an expert NLP evaluator. Please evaluate the sentiment of the following text:"

MODEL = "gpt-3.5-turbo-16k"
TEMPERATURE = 0.0
MAX_TOKENS = 300

@retry(stop=stop_after_attempt(max_attempts), wait=wait_fixed(wait_time), after=after_log(logger, logging.ERROR))
def classify(input_string:str):
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{input_string}"}
        ]
    response = openai.ChatCompletion.create(
                model=MODEL,
                api_key=api_key,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
                functions=functions,
                function_call={"name": "print_sentiment"}  
            )
    function_call = response.choices[0].message["function_call"]
    argument = json.loads(function_call["arguments"])
    return list(argument.values())[0]


def main():
    api = wandb.Api()
    
    df = pd.DataFrame(columns=['country', 'question_type', 'question_template', 'response',
        'blinded_response', 'trial', 'elapsed_time', "system_country"])

    api = wandb.Api()

    c = "dubai"
    tables_dict = {
        # "abu_dhabi": ["run-0ep0legf-abu_dhabi_5k", "abu_dhabi_5k"],
        # "mexico": ["run-fwel09o1-mexico_28k", "mexico_2-8k"],
        # "ireland": ["run-80pf5bhm-ireland_28k", "ireland_2-8k"],
        # "saudi_arabia": ["run-urgxrrlt-saudi_arabia_28k", "saudi_arabia_2-8k"],
        # "usa": ["run-y4o7qg5r-us_28k", "us_2-8k"],
        # "china": ["run-epxnaejk-china_28k", "china_2-8k"],
        # "uk": ["run-lsgzp9pj-uk_28k", "uk_2-8k"],
        # "japan": ["run-wzjavac5-japan_28k", "japan_2-8k"],
        "dubai": ["run-0mtnhirf-dubai_28k", "dubai_2-8k"],
    }

    for sys_country in tables_dict.keys():
        table_id = tables_dict[sys_country]
        print(table_id)
        artifact = api.artifact(f"morgan/llm-country-preference/{table_id[0]}:v0")
        tmp_df = artifact.get(table_id[1]).get_dataframe()
        tmp_df["system_country"] = sys_country
        df = pd.concat([df, tmp_df])
    print(len(df))

    # filter empty responses from the LLM
    filtered_df = df.query("response != ''")
    print(len(filtered_df))

    b_responses_ls = list(filtered_df.blinded_response.values)  #[:10]
    # filtered_df = filtered_df.iloc[:10,:]
    
    n_procs = 4
    out_ls = []
    for i in tqdm(range(0, len(b_responses_ls), n_procs)):
        end_index = min(i+n_procs, len(b_responses_ls)) # Make sure we don't go past the end of the list
        inp_ls = b_responses_ls[i:end_index]
        out = parallel(classify, inp_ls, n_workers=n_procs)
        out_ls.append(out)  # Save the segment in the output list

    # unpack list of lists
    fin_ls = [i for o in out_ls for i in o]
    print(len(fin_ls))

    tmp_df = pd.DataFrame({"sentiment": fin_ls})
    tmp_df.to_csv(f"tmp_{c}_sentiment2.csv")

    filtered_df["sentiment"] = fin_ls
    filtered_df.to_csv(f"{c}_sentiment_filtered.csv")
    wandb.init(project="llm-country-preference", 
               name=f"{c}_sentiment", 
               config={"model":MODEL,
                       "temperature":TEMPERATURE,
                       "max_tokens": MAX_TOKENS,
                       "system_country":c,
                       "openai_function": functions[0],
                       "system_prompt": SYSTEM_PROMPT,

                       },
               tags=[f"{c}"])
    wandb.log({f"{c}_labelled": filtered_df})

if __name__ == '__main__':
    main()