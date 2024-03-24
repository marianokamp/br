import json
import os
from pprint import pprint

from config import Config
from monitor_br import create_runs, generate

conf = Config.from_yaml_file("config.yaml")
runs = create_runs(conf)
print("one", runs[0])

results = generate(**runs[0])
pprint(results)
print("done")


def lambda_handler(event, context):
    json_region = os.environ["AWS_REGION"]
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"Region ": json_region}),
    }
