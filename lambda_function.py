import json
import os
import random
from pprint import pprint
from botocore.exceptions import ClientError, EventStreamError  # , EventStreamError

from config import Config
from monitor_br import create_runs, run_and_report, report_run

conf = Config.from_yaml_file("config.yaml")
runs = create_runs(conf)
print("one", runs[0])

#results = generate(**runs[0])
#pprint(results)
print("done init")


def lambda_handler(event, context):
    json_region = os.environ["AWS_REGION"]

    run = random.choice(runs)
    try:
        result = run_and_report(conf, run)
        print('result:\n``', result)
        report_run(run, result)
    
    except ClientError as err:
        print("err", err)
        print(run["scenario"], run["model_id"], run["region"])
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "text/html"},
            "body": f'Trouble executing run: {run}, ended with error: {str(err)}'  
        }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"Region ": json_region}),
    }
